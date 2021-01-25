

class Freeze:

    def __init__(self):
        self._content = list()
        self._cnt = 0

    def put(self, something):
        self._content.append(something)
        self._cnt = self._cnt + 1

    def look(self):
        print(self._content)
        print(self._cnt)

    def take(self, something):
        try:
            self._content.remove(something)
            print('Take a ', something)
            self._cnt = self._cnt - 1
            return something
        except ValueError as err:
            print('No this item')



def main():
    my_freeze = Freeze()

    my_freeze.put('Торт')
    my_freeze.look()

    my_freeze.put('Пиво')
    my_freeze.look()

    my_freeze.take('Торт')
    my_freeze.look()

    my_freeze.take('Торт')
    my_freeze.look()

    neighbour_freeze = Freeze()

    neighbour_freeze.put('Салат')
    neighbour_freeze.put('Салат')

    print()


if __name__ == '__main__':
    main()