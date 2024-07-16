import pandas as pd

df = pd.read_csv("profit_and_loss_report.csv")


# Define the categories
payroll_categories = [
    '70110 Employee Wages', '70120 Bonuses', '70131 Health Insurance',
    '70132 Shareholder Life Insurance', '70133 401(k) Match',
    '70140 Payroll Taxes', '70150 Payroll / 401K Fees', '70160 Workman\'s Comp insurance'
]

expensify_categories = [
    '62100 Client Development', '62150 Meals - Client/Vendor', '70121 Air travel',
    '70122 Lodging', '70123 Ground Transportation', '70220 Meals - employee travel',
    '70230 Entertainment (non-food)'
]

billcom_categories = [
    '51000 COGS - Labor', '52000 COGS - Non-Labor', '53000 COGS - Research (Products)',
    '54000 COGS - Shipping', '56000 Reimbursed Expenses', '70310 Accounting/Tax',
    '70320 Legal Fees', '70340 IT support', '70350 General Consultant', '70410 Office Rent',
    '70420 Utilities/Trash services', '70430 Office Cleaning / Repairs', '70440 Telephone / Internet',
    '70450 Office Supplies/Equipment', '70500 Online Software & Subscriptions', '70525 Office Events',
    '70530 Office Meals/ Snacks', '70535 Recruiting/ HR', '70540 Memberships/Dues', '70545 Employee Retention',
    '70550 Professional Development', '70600 Bank & Merchant Fees', '70650 Local taxes /Registrations',
    '70675 Auto expense', '70700 Business Insurance'
]

def calculate_sum(categories, df):
    return df[df['Subcategory'].isin(categories)]['Amount'].sum()

# Calculate sums
ap_payroll_sum = calculate_sum(payroll_categories, df)
ap_expensify_sum = calculate_sum(expensify_categories, df)
ap_billcom_sum = calculate_sum(billcom_categories, df)

print(f'Payroll Total: ${ap_payroll_sum}')
print(f'Expensify Total: ${ap_expensify_sum}')
print(f'Bill.com Total: ${ap_billcom_sum}')