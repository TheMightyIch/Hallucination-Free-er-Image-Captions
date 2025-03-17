from Models.AbstractModel import AbstractModel


class test (AbstractModel):
    def __init__(self, model_name: str ):
        super().__init__(model_name)
        self.banana="apple"

    def generateModel(self, **inputs):
        print(self.banana)
        return self.banana

    def generateResponse(self, messages):
        print(self.banana)
        return messages

if __name__ == "__main__":
    test = test("test")
    print(test.generateModel())
    print(test.generateResponse("Hello, how are you?"))