(*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *)

open Ast

type coverage_data = {
  expression: Expression.t option;
  type_: Type.t;
}
[@@deriving compare, sexp, show, hash, to_yojson]

type typeisany =
  | ParameterIsAny
  | OtherExpressionIsAny
[@@deriving compare, sexp, show, hash, to_yojson]

type reason =
  | TypeIsAny of typeisany
  | ContainerParameterIsAny
  | CallableParameterIsUnknownOrAny
  | CallableReturnIsAny
[@@deriving compare, sexp, show, hash, to_yojson]

type coverage_gap = {
  coverage_data: coverage_data;
  reason: reason;
}
[@@deriving compare, sexp, show, hash, to_yojson]

type coverage_gap_by_location = {
  location: Location.t;
  function_name: string option;
  type_: Type.t;
  reason: string list;
}
[@@deriving equal, compare, sexp, show, hash, to_yojson]

type coverage_for_path = {
  total_expressions: int;
  coverage_gaps: coverage_gap_by_location list;
}
[@@deriving compare, sexp, show, hash, to_yojson]

type hover_info = {
  value: string option;
  docstring: string option;
}
[@@deriving sexp, show, compare, yojson { strict = false }]

type coverage_data_lookup

val create_of_module : TypeEnvironment.ReadOnly.t -> Reference.t -> coverage_data_lookup

val get_coverage_data
  :  coverage_data_lookup ->
  position:Location.position ->
  (Location.t * coverage_data) option

val get_all_nodes_and_coverage_data : coverage_data_lookup -> (Location.t * coverage_data) list

type symbol_with_definition =
  | Expression of Expression.t
  | TypeAnnotation of Expression.t
[@@deriving compare, show]

type cfg_data = {
  define_name: Reference.t;
  node_id: int;
  statement_index: int;
}
[@@deriving compare, show]

type symbol_and_cfg_data = {
  symbol_with_definition: symbol_with_definition;
  cfg_data: cfg_data;
  use_postcondition_info: bool;
}
[@@deriving compare, show]

val location_insensitive_compare_symbol_and_cfg_data
  :  symbol_and_cfg_data ->
  symbol_and_cfg_data ->
  int

val narrowest_match : symbol_and_cfg_data list -> symbol_and_cfg_data option

val find_narrowest_spanning_symbol
  :  type_environment:TypeEnvironment.ReadOnly.t ->
  module_reference:Reference.t ->
  Location.position ->
  symbol_and_cfg_data option

val resolve_definition_for_symbol
  :  type_environment:TypeEnvironment.ReadOnly.t ->
  module_reference:Reference.t ->
  symbol_and_cfg_data ->
  Location.WithModule.t option

val location_of_definition
  :  type_environment:TypeEnvironment.ReadOnly.t ->
  module_reference:Reference.t ->
  Location.position ->
  Location.WithModule.t option

val resolve_completions_for_symbol
  :  type_environment:TypeEnvironment.ReadOnly.t ->
  symbol_and_cfg_data ->
  AttributeResolution.AttributeDetail.t list

val completion_info_for_position
  :  type_environment:TypeEnvironment.ReadOnly.t ->
  module_reference:Reference.t ->
  Location.position ->
  AttributeResolution.AttributeDetail.t list

val resolve_type_for_symbol
  :  type_environment:TypeEnvironment.ReadOnly.t ->
  symbol_and_cfg_data ->
  Type.t option

val classify_coverage_data : coverage_data -> coverage_gap option

val coverage_gaps_in_module : coverage_data list -> coverage_gap list

val get_expression_level_coverage : coverage_data_lookup -> coverage_for_path

val hover_info_for_position
  :  type_environment:TypeEnvironment.ReadOnly.t ->
  module_reference:Reference.t ->
  Location.position ->
  hover_info

val parameter_is_any_message : string list

val expression_is_any_message : string list

val container_parameter_is_any_message : string list

val callable_parameter_is_unknown_or_any_message : string list

val callable_return_is_any_message : string list
