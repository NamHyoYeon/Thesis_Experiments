import pandas as pd
import matplotlib.pyplot as plt

file_path = './data/Online Retail Dataset.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    print(df.info())

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    plt.plot(x, y)
    plt.title("My Plot")
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.show()
