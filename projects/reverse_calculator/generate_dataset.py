import csv
import random

def calculate(num1, num2, opr):
    if opr==1:
        data = [num1, num2, num1+num2, 1]
    elif opr==2:
        data = [num1, num2, num1-num2, 2]
    elif opr==3:
        data = [num1, num2, num1*num2, 3]
    elif opr==4:
        if num2==0:
            data = [num1, num2, -99999, 0]
        else:
            data = [num1, num2, num1/num2, 4]
    else:
        if num2==0:
            data = [num1, num2, -99999, 0]
        else:
            data = [num1, num2, num1%num2, 5]
    return data

def generate_dataset():
    dataset = []
    for i in range(500):
        opr = random.randint(1, 2)
        num1 = random.randint(-500, 500)
        num2 = random.randint(-500, 500)
        result = calculate(num1, num2, opr)
        dataset.append(result)
    return dataset


def main():
    dataset = generate_dataset()
    with open('dataset/reverse_1-2_dataset.data', 'w', newline='') as dataset_file:
        wr = csv.writer(dataset_file)
        wr.writerow(['num1','num2','result','class'])
        wr.writerows(dataset)

main()
