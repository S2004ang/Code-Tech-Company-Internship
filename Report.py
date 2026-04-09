import pandas as pd

# Read data
data = pd.read_csv("data.csv")

# Analysis
total_students = len(data)
average_marks = data["Marks"].mean()
max_marks = data["Marks"].max()
min_marks = data["Marks"].min()

print("Analysis Done")



from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Create PDF
pdf = canvas.Canvas("report.pdf", pagesize=letter)

# Title
pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(200, 750, "Student Report")

# Content
pdf.setFont("Helvetica", 12)
pdf.drawString(100, 700, f"Total Students: {total_students}")
pdf.drawString(100, 680, f"Average Marks: {average_marks:.2f}")
pdf.drawString(100, 660, f"Max Marks: {max_marks}")
pdf.drawString(100, 640, f"Min Marks: {min_marks}")

# Save PDF
pdf.save()

print("PDF Generated Successfully")