"""Generate a sample sales CSV (same schema as the original tutorial)."""
import csv
rows = [["Date","Product","Units Sold","Revenue"],
 ["2025-08-01","Widget A",10,250],["2025-08-02","Widget B",5,125],
 ["2025-08-03","Widget A",7,175],["2025-08-04","Widget C",3,90],["2025-08-05","Widget B",8,200]]
with open("sales_data.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)
print("sales_data.csv created")
