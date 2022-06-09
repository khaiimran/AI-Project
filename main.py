import driver

def main():
    template = 'logo_train.png'
    choice = input("Enter 0 to upload image, otherwise it will run using camera: ")
    if choice != 0:
        driver.cam(template)
    else:
        image = 'test.apng'
        driver.img(template, image)

if __name__ == '__main__':
    main()