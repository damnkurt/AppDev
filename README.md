Install all the dependencies in your virtual environment .venv

Make an .env file
GOOGLE_API_KEY=your key here

make a database name it insurance_ai, make 2 tables for accounts and admins with 
Fields: name (varchar), email (varchar), age (int), sex (varchar - 'Male' or 'Female'), 
            status (int - 0=inactive, 1=active), hashed_password (varchar), 
            time_created (timestamp), time_modified (timestamp on update)


