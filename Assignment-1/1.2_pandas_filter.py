import pandas as pd

data = {
    "Student Name": ["Rahul", "Amit", "Priya", "Sneha", "Rohan"],
    "Roll Number": [101, 102, 103, 104, 105],
    "Marks": [85, 72, 91, 78, 88],
    "Attendance": [90, 85, 95, 80, 92]
}

df = pd.DataFrame(data)

print("Students scoring above 80 marks:")
print(df[df["Marks"] > 80])
