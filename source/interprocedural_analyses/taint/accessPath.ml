(*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *)

(* AccessPath:
 *
 * Defines an access path, which is a subset of python expressions.
 *
 * Access paths are used to represent the model of a callable, which describes
 * the set of sources produced by the callable, and the set of sinks reachable
 * from the parameters of the callable.
 *
 * This also implements the logic to match actual parameters at call sites to
 * the formal parameters of a callable.
 *)

open Core
open Ast
open Expression

(** Roots representing parameters, locals, and special return value in models. *)
module Root = struct
  module T = struct
    type t =
      | LocalResult (* Special root representing the return value location. *)
      | PositionalParameter of {
          position: int;
          name: Identifier.t;
          positional_only: bool;
        }
      | NamedParameter of { name: Identifier.t }
      | StarParameter of { position: int }
      | StarStarParameter of { excluded: Identifier.t list }
      | Variable of Identifier.t
      | CapturedVariable of Identifier.t
    [@@deriving compare, eq, hash, sexp, show { with_path = false }]

    let chop_parameter_prefix name =
      match String.chop_prefix ~prefix:"$parameter$" name with
      | Some chopped -> chopped
      | None -> name


    let parameter_name = function
      | PositionalParameter { name; _ }
      | NamedParameter { name } ->
          Some name
      | StarParameter _ -> Some "*"
      | StarStarParameter _ -> Some "**"
      | _ -> None


    let pp_external formatter = function
      | LocalResult -> Format.fprintf formatter "result"
      | PositionalParameter { position = _; name; _ } -> Format.fprintf formatter "formal(%s)" name
      | NamedParameter { name } -> Format.fprintf formatter "formal(%s)" name
      | StarParameter { position } -> Format.fprintf formatter "formal(*rest%d)" position
      | StarStarParameter _ -> Format.fprintf formatter "formal(**kw)"
      | Variable name -> Format.fprintf formatter "local(%s)" name
      | CapturedVariable name -> Format.fprintf formatter "captured_variable(%s)" name


    let show_external = Format.asprintf "%a" pp_external

    let pp_internal = pp

    let show_internal = Format.asprintf "%a" pp_internal

    let variable_to_captured_variable = function
      | Variable name -> CapturedVariable name
      | root -> root


    let captured_variable_to_variable = function
      | CapturedVariable name -> Variable name
      | root -> root


    let is_captured_variable = function
      | CapturedVariable _ -> true
      | _ -> false
  end

  include T
  module Set = Stdlib.Set.Make (T)
end

module NormalizedParameter = struct
  type t = {
    root: Root.t;
    (* Qualified name (prefixed with `$parameter$`), ignoring stars. *)
    qualified_name: Identifier.t;
    original: Parameter.t;
  }
end

let normalize_parameters parameters =
  let normalize_parameters
      position
      (seen_star, excluded, normalized)
      ({ Node.value = { Parameter.name = qualified_name; _ }; _ } as original)
    =
    if Identifier.equal (Identifier.sanitized qualified_name) "/" then
      let mark_as_positional_only ({ NormalizedParameter.root; _ } as parameter) =
        let root =
          match root with
          | Root.PositionalParameter { position; name; _ } ->
              Root.PositionalParameter { position; name; positional_only = true }
          | _ -> root
        in
        { parameter with root }
      in
      seen_star, excluded, List.map ~f:mark_as_positional_only normalized
    else if String.is_prefix ~prefix:"**" qualified_name then
      let qualified_name = String.chop_prefix_exn qualified_name ~prefix:"**" in
      ( true,
        excluded,
        { NormalizedParameter.root = Root.StarStarParameter { excluded }; qualified_name; original }
        :: normalized )
    else if Identifier.equal (Identifier.sanitized qualified_name) "*" then
      (* This is not a real starred argument, but instead is just the indicator for the beginning
       * of the keyword only arguments *)
      true, excluded, normalized
    else if String.is_prefix ~prefix:"*" qualified_name then
      let qualified_name = String.chop_prefix_exn qualified_name ~prefix:"*" in
      ( true,
        excluded,
        { NormalizedParameter.root = Root.StarParameter { position }; qualified_name; original }
        :: normalized )
    else if seen_star then
      let unqualified_name = Root.chop_parameter_prefix qualified_name in
      ( true,
        unqualified_name :: excluded,
        {
          NormalizedParameter.root = Root.NamedParameter { name = unqualified_name };
          qualified_name;
          original;
        }
        :: normalized )
    else
      let unqualified_name = Root.chop_parameter_prefix qualified_name in
      ( false,
        unqualified_name :: excluded,
        {
          NormalizedParameter.root =
            Root.PositionalParameter { position; name = unqualified_name; positional_only = false };
          qualified_name;
          original;
        }
        :: normalized )
  in
  List.foldi parameters ~f:normalize_parameters ~init:(false, [], [])
  |> fun (_, _, parameters) -> List.rev parameters


module Path = struct
  type t = Abstract.TreeDomain.Label.t list

  let pp = Abstract.TreeDomain.Label.pp_path

  let show = Abstract.TreeDomain.Label.show_path

  let compare = Abstract.TreeDomain.Label.compare_path ?cmp:None

  let equal = Abstract.TreeDomain.Label.equal_path

  let empty = []

  let is_prefix = Abstract.TreeDomain.Label.is_prefix
end

type argument_match = {
  root: Root.t;
  actual_path: Path.t;
  formal_path: Path.t;
}
[@@deriving compare, show { with_path = false }]

type argument_position =
  [ `Precise of int
  | `Approximate of int
  ]

(* Will preserve the order in which the arguments were matched to formals. *)
let match_actuals_to_formals arguments roots =
  let open Root in
  let increment = function
    | `Precise n -> `Precise (n + 1)
    | `Approximate n -> `Approximate (n + 1)
  in
  let approximate = function
    | `Precise n -> `Approximate n
    | `Approximate n -> `Approximate n
  in
  let filter_to_named matched_names name_or_star_star formal =
    match name_or_star_star, formal with
    | `Name actual_name, (NamedParameter { name } | PositionalParameter { name; _ })
      when Identifier.equal actual_name name ->
        Some { root = formal; actual_path = []; formal_path = [] }
    | `Name actual_name, StarStarParameter { excluded }
      when not (List.exists ~f:(Identifier.equal actual_name) excluded) ->
        let index = Root.chop_parameter_prefix actual_name in
        Some
          {
            root = formal;
            actual_path = [];
            formal_path = [Abstract.TreeDomain.Label.create_name_index index];
          }
    | `StarStar, NamedParameter { name }
    | `StarStar, PositionalParameter { name; _ }
      when not (Base.Set.mem matched_names name) ->
        Some
          { root = formal; actual_path = [Abstract.TreeDomain.Label.Index name]; formal_path = [] }
    | `StarStar, StarStarParameter _ ->
        (* assume entire structure is passed. We can't just pick all but X fields. *)
        Some { root = formal; actual_path = []; formal_path = [] }
    | (`Name _ | `StarStar), _ -> None
  in
  let filter_to_positional position formal =
    let position = (position :> [ `Star of argument_position | argument_position ]) in
    match position, formal with
    | `Precise actual_position, PositionalParameter { position; _ } when actual_position = position
      ->
        Some { root = formal; actual_path = []; formal_path = [] }
    | `Approximate minimal_position, PositionalParameter { position; _ }
      when minimal_position <= position ->
        Some { root = formal; actual_path = []; formal_path = [] }
    | `Star (`Precise actual_position), PositionalParameter { position; _ }
      when actual_position <= position ->
        Some
          {
            root = formal;
            actual_path = [Abstract.TreeDomain.Label.create_int_index (position - actual_position)];
            formal_path = [];
          }
    | `Star (`Approximate minimal_position), PositionalParameter { position; _ }
      when minimal_position <= position ->
        Some { root = formal; actual_path = [Abstract.TreeDomain.Label.AnyIndex]; formal_path = [] }
    | `Precise actual_position, StarParameter { position } when actual_position >= position ->
        Some
          {
            root = formal;
            actual_path = [];
            formal_path = [Abstract.TreeDomain.Label.create_int_index (actual_position - position)];
          }
    | `Approximate _, StarParameter _ ->
        (* Approximate: We can't filter by minimal position in either direction here. Think about
           the two following cases:

           1. ```def f(x, y, *z): ... f( *[1, 2], approx) ``` In this case, even though the minimal
           position for approx is < the starred parameter, it will match to z.

           2. ```def f( *z): ... f( *[], 1, approx) ``` In this case, we'll have approx, which has a
           minimal position > the starred parameter, flow to z. *)
        Some { root = formal; actual_path = []; formal_path = [Abstract.TreeDomain.Label.AnyIndex] }
    | `Star _, StarParameter _ ->
        (* Approximate: can't match up ranges, so pass entire structure *)
        Some { root = formal; actual_path = []; formal_path = [] }
    | (`Star _ | `Precise _ | `Approximate _), _ -> None
  in
  let match_actual matched_names (position, matches) ({ Call.Argument.name; value } as argument) =
    match name, value.Node.value with
    | None, Starred (Once _) ->
        let formals = List.filter_map roots ~f:(filter_to_positional (`Star position)) in
        approximate position, (argument, formals) :: matches
    | None, Starred (Twice _) ->
        let formals = List.filter_map roots ~f:(filter_to_named matched_names `StarStar) in
        position, (argument, formals) :: matches
    | None, _ ->
        let formals = List.filter_map roots ~f:(filter_to_positional position) in
        increment position, (argument, formals) :: matches
    | Some { value = name; _ }, _ ->
        let unqualified_name = chop_parameter_prefix name in
        let formals =
          List.filter_map roots ~f:(filter_to_named matched_names (`Name unqualified_name))
        in
        position, (argument, formals) :: matches
  in
  let matched_names =
    let matched_named_arguments =
      List.map arguments ~f:(fun { Call.Argument.name; _ } -> name)
      |> List.filter_opt
      |> List.map ~f:Node.value
      |> List.map ~f:Identifier.sanitized
    in
    let matched_positional_arguments =
      let matched_positions =
        let unstarred_positional = function
          | { Call.Argument.name = Some _; _ }
          (* We deliberately ignore starred arguments here, and pessimistically assume they match 0
             parameters. *)
          | { Call.Argument.name = None; value = { Node.value = Starred _; _ } } ->
              false
          | { Call.Argument.name = None; _ } -> true
        in
        List.count arguments ~f:unstarred_positional
      in
      let get_matched_root = function
        (* Positions are 0 indexed, hence the < instead of <=. *)
        | PositionalParameter { position; name; _ } when position < matched_positions -> Some name
        | _ -> None
      in
      List.filter_map roots ~f:get_matched_root
    in
    String.Set.of_list (matched_named_arguments @ matched_positional_arguments)
  in

  let _, result = List.fold arguments ~f:(match_actual matched_names) ~init:(`Precise 0, []) in
  List.rev result


type t = {
  root: Root.t;
  path: Path.t;
}
[@@deriving compare]

let pp formatter { root; path } = Format.fprintf formatter "%a%a" Root.pp_external root Path.pp path

let show access_path = Format.asprintf "%a" pp access_path

let create root path = { root; path }

let extend { root; path = original_path } ~path = { root; path = original_path @ path }

let get_index expression =
  match Interprocedural.CallResolution.extract_constant_name expression with
  | Some "True" -> Abstract.TreeDomain.Label.Index "1"
  | Some "False" -> Abstract.TreeDomain.Label.Index "0"
  | Some name -> Abstract.TreeDomain.Label.Index name
  | None -> Abstract.TreeDomain.Label.AnyIndex


let to_json { root; path } = `String (Format.asprintf "%a%a" Root.pp_external root Path.pp path)

let of_expression expression =
  let rec of_expression path = function
    | { Node.value = Expression.Name (Name.Identifier identifier); _ } ->
        Some { root = Root.Variable identifier; path }
    | { Node.value = Name (Name.Attribute { base; attribute; _ }); _ } ->
        let path = Abstract.TreeDomain.Label.Index attribute :: path in
        of_expression path base
    | {
        Node.value =
          Call
            {
              Call.callee =
                {
                  Node.value =
                    Name (Name.Attribute { base; attribute = "__getitem__"; special = true });
                  _;
                };
              arguments =
                [
                  {
                    Call.Argument.name = None;
                    value = { Node.value = Expression.Constant (Constant.String { value; _ }); _ };
                  };
                ];
            };
        _;
      } ->
        let path = Abstract.TreeDomain.Label.Index value :: path in
        of_expression path base
    | _ -> None
  in
  of_expression [] expression


let dictionary_keys = Abstract.TreeDomain.Label.Field "**keys"
