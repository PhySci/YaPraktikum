import pandas as pd


class Employee:

    def __init__(self, name: str ='Unnamed', date_birth: str = None, position='stableman', grade: int =0):
        self.name = name
        self.date_birth = date_birth
        self.position = position
        self.grade = grade

    def promote(self):
        self.grade = self.grade + 1

    def show_info(self):
        for k, v in self.__dict__.items():
            print(k, v)

    def __str__(self):
        s = ' '.join([str(k) +':'+ str(v) + ', ' for k, v in self.__dict__.items()])
        return s


def main():
    empl1 = Employee(name='Petr', date_birth='1_04_1990', position='postdoc', grade=1)
    empl1.show_info()
    empl1.promote()
    empl1.show_info()


if __name__ == '__main__':
    main()

