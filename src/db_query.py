import psycopg2

# Establish connection to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="telecom_customer_intelligence_hub",
    user="postgres",
    password="123"
)

# Create a cursor object
cur = conn.cursor()

# Write a SQL query
query = "SELECT * FROM public.xdr_data LIMIT 10;"

# Execute the query
cur.execute(query)

# Fetch the results
rows = cur.fetchall()

# Print the results
for row in rows:
    print(row)

# Close the cursor and connection
cur.close()
conn.close()