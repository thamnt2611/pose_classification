class ClassTracker(object):
    def __init__(self, name):
        self.class_dict = {}
    
    def register_class(self, cls):
        cls_name = cls.__name__
        self.class_dict[cls_name] = cls

ANM = ClassTracker("Animal")

@ANM.register_class
class Dog(object):
    pass

if __name__ == "__main__":
    print(ANM.class_dict)