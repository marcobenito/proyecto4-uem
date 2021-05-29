import json
import base64
import os

print(os.getcwd())
print(os.listdir())

filename = '9'

def img2json(filename):
    data = {}
    # Read file
    with open(filename + '.jpg', mode='rb') as file:
        img = file.read()

    # Encode file
    data['img'] = base64.b64encode(img).decode('utf-8')

    # Write json file
    with open(filename + '.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    img2json(filename)