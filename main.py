import driver

def main():
    template = 'logo_train.png'
    choice = int (input())
    if choice != 0:
        driver.cam(template)
    else:
        image = 'test.apng'
        driver.img(template, image)

if __name__ == '__main__':
    main()