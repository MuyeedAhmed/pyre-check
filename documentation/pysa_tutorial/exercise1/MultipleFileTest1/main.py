from animal import Animal
from dog import Dog
from cat import Cat

def main():
    dog_name = input("Name: ")
    my_dog = Dog(dog_name)
    print(my_dog.speak())
    
    cat_name = input("Name: ")
    my_cat = Cat(cat_name)
    my_cat_real_age = int(input("Age: "))
    print(my_cat.age(my_cat_real_age))    
    print(my_cat.status(my_cat_real_age))
    
    animal_name = input("Name: ")
    my_animal = Animal(animal_name)
    print(my_animal.speak())
    
    

if __name__ == "__main__":
    main()
