from animal import Animal

class Cat(Animal):
    def speak(self):
        return f"{self.name} says meaw!"
    
    def age(self, real_age):
        return real_age * 7

    def status(self, real_age):
        dead=False
        if real_age > 3:
            print(f"Cat {real_age} is dead")
            dead = True
        else:
            print("Cat is alive")
        return dead