import streamlit as st
import db_utils as db
import pandas as pd
import random
import requests
# from sklearn.tree import DecisionTreeClassifier, export_text 
from xgboost import XGBClassifier
from google import genai
from dotenv import load_dotenv
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix,precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def handle_admin_query(user_input, client):
    """Handle admin mode SQL queries"""
    prompt = f"""
    You are a SQL assistant. Convert the user request into a valid SQL query.
    Database: MySQL
    Table: 'accounts'
    Fields: name (varchar), email (varchar), age (int), sex (varchar - 'Male' or 'Female'), 
            status (int - 0=inactive, 1=active), hashed_password (varchar - don't show), 
            time_created (timestamp), time_modified (timestamp)
    
    Instructions:
    - Output ONLY the SQL query, nothing else
    - No explanations, comments, or markdown
    - Use proper SQL syntax
    - Only SELECT queries are allowed
    
    User request: {user_input}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        if response.text is None:
            st.error("❌ No response from AI")
            return
            
        sql_query = response.text.strip()
        
        # Safety check
        dangerous_keywords = ["drop", "truncate", "alter", "rename", "create", 
                            "grant", "revoke", "insert", "update", "delete", 
                            "modify", "replace"]
        
        if any(keyword in sql_query.lower() for keyword in dangerous_keywords):
            error_msg = f"❌ Blocked potentially dangerous query: {sql_query}"
            st.session_state.history["Admin"].append(("Error", error_msg))
            st.error("⚠️ This type of query is not allowed for safety reasons.")
            return
        
        # Show the SQL query
        st.session_state.history["Admin"].append(("AI (SQL Query)", sql_query))
        st.code(sql_query, language="sql")
        
        # Execute query if it's a SELECT
        if sql_query.lower().startswith("select"):
            execute_sql_query(sql_query)
        else:
            st.warning("Only SELECT queries can be executed automatically.")
            
    except Exception as e:
        error_msg = f"❌ Error generating query: {str(e)}"
        st.session_state.history["Admin"].append(("Error", error_msg))
        st.error("Failed to process your request. Please try again.")

def execute_sql_query(sql_query):
    """Execute SQL query and display results"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        results = cursor.fetchall()
        columns = [i[0] for i in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        
        st.session_state.history["Admin"].append(("Results", df))
        st.write("**Query Results:**")
        st.dataframe(df)
        
        # Show result summary
        if len(df) > 0:
            st.success(f"✅ Found {len(df)} record(s)")
        else:
            st.info("No records found")
            
    except Exception as e:
        error_msg = f"❌ SQL Error: {str(e)}"
        st.session_state.history["Admin"].append(("Error", error_msg))
        st.error(f"Database error: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def handle_normal_chat(user_input, client):
    """Handle normal chat mode"""
    prompt = f"""
    You are a helpful assistant. Provide a friendly, concise response to the user's message.
    Keep it conversational and not too long.
    
    User message: {user_input}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        ai_text = response.text if response.text else "I apologize, but I couldn't generate a response."
        st.session_state.history["Normal"].append(("AI", ai_text))
        st.write(ai_text)
        
    except Exception as e:
        error_msg = f"❌ Chat error: {str(e)}"
        st.session_state.history["Normal"].append(("Error", error_msg))
        st.error("Sorry, I encountered an error. Please try again.")

def handle_user_chat_query(user_input, client):
            """Handle user chatbot queries with driving safety focus"""
            
            driving_context = """
            You are a helpful driving safety assistant. Your expertise includes:
            - Driving risk factors and prevention
            - Road safety tips and best practices
            - Weather and road condition advice
            - Fatigue management while driving
            - Vehicle maintenance recommendations
            - Defensive driving techniques
            - Traffic rules and regulations
            
            Provide helpful, accurate, and concise advice. If the question is not related to driving safety,
            politely steer the conversation back to driving topics or explain why driving safety is important.
            
            Keep responses short, conversational and practical.
            """
            
            prompt = f"""
            {driving_context}
            
            User Question: {user_input}
            
            Please provide a helpful response focused on driving safety:
            """
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                return response.text if response.text else "I'm not sure how to help with that. Could you ask about driving safety instead?"
                
            except Exception as e:
                return "I apologize, but I'm having trouble responding right now. Please try asking about driving safety tips or risk factors."

def fetch_geolocation(ip=None):
    if ip:
        url = f"http://ip-api.com/json/{ip}"
    else:
        # Auto-detect IP
        url = "http://ip-api.com/json/"
    
    response = requests.get(url, headers={"Accept": "application/json"})
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "success":
            return data.get("query"), data.get("city"), data.get("lat"), data.get("lon")
        else:
            return None, None, None, None
    else:
        return None, None, None, None

# ---------------- Session State ----------------
if "auth" not in st.session_state:
    st.session_state.auth = None   # None, "admin", "user"
if "email" not in st.session_state:
    st.session_state.email = ""


# ---------------- Current Route ----------------
route = st._get_query_params().get("page", ["login"])[0]

# ---------------- Login Page ----------------
if route == "login" and st.session_state.auth is None:
    st.set_page_config(
        page_title="Login",
        page_icon="🔐",
        layout="centered"
    )
    st.image("Images/logo.png")
    st.title("🔑 Login Page")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    choicecol, space,col1, col2 = st.columns([1,3.5,0.65,0.8])
    choice = choicecol.selectbox("Log in as",["User","Admin"])
    
    
    try:
        if col1.button("Login"):
            if choice == "Admin":
                conn = db.get_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM admins WHERE email=%s", (email,))
                result = cursor.fetchone()
                hashed = result['hashed_password']
                if db.check_password(password,hashed):
                    if db.check_admin(email, password):
                        st.session_state.auth = "admin"
                        st.session_state.email = email
                        st._set_query_params(page="admin_ui")
                        st.rerun()
                    else:
                        st.warning("Wrong Credentials")
                else:
                    st.warning("Wrong password")
            elif choice == "User":
                conn = db.get_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM accounts WHERE email=%s", (email,))
                result = cursor.fetchone()
                hashed = result['hashed_password']
                if db.check_password(password,hashed):
                    if db.check_user(email, password):
                        st.session_state.auth = "user"
                        st.session_state.email = email
                        st._set_query_params(page="user_ui")
                        st.rerun()
                    else:
                        st.warning("Wrong Credentials")
                else:
                    st.warning("Wrong password")
        if col2.button("Sign Up"):
            st.session_state.auth = "signup"
            st.session_state.email = email
            st._set_query_params(page="sign_up")
            st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")
        st.rerun()
        

# ---------------- Admin Dashboard ----------------
elif route == "admin_ui" and st.session_state.auth == "admin":
    st.sidebar.image("Images/logo.png")
    
    st.set_page_config(
        page_title="Admin Dashboard",
        page_icon="👤",
        layout="wide",  # "centered" or "wide"
        initial_sidebar_state="collapsed"  # "collapsed" or "expanded"
    )

    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM admins where email = \"{st.session_state.email}\"")
    results = cursor.fetchone()
    name = results[0]
    
    st.title("🛠 Admin Dashboard")
    st.write(f"Welcome Admin: **{name[0].upper() + name[1:] }**")

    crudcol, chatbot = st.columns([3,1])
    with crudcol:
        crud_action = st.selectbox("Action", ["Create User", "Read Users", "Update User", "Delete User"])

        conn = db.get_connection()
        cursor = conn.cursor(dictionary=True)

        if crud_action == "Create User":
            with st.form("create_user"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                age = st.number_input("Age", min_value=18, max_value=100)
                sex = st.selectbox("Sex", ["Male", "Female"])
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Create")
                if submitted:
                    success, message = db.register_user(name, email, age, sex, password)
                    if success:
                        st.success("✅ User created successfully!")
                    else:
                        st.error(f"❌ {message}")

        elif crud_action == "Read Users":
            cursor.execute("SELECT name, email, age, sex, status, time_created, time_modified FROM accounts")
            results = cursor.fetchall()
            df = pd.DataFrame(results)
            st.dataframe(df)

        elif crud_action == "Update User":
            email_to_update = st.text_input("Enter user email to update")
            new_status = st.selectbox("New Status", [0, 1], format_func=lambda x: "Inactive" if x == 0 else "Active")
            if st.button("Update"):
                cursor.execute("UPDATE accounts SET status=%s, time_modified=NOW() WHERE email=%s", (new_status, email_to_update))
                conn.commit()
                st.success("✅ User updated successfully!")

        elif crud_action == "Delete User":
            email_to_delete = st.text_input("Enter user email to delete")
            if st.button("Delete"):
                cursor.execute("DELETE FROM accounts WHERE email=%s", (email_to_delete,))
                conn.commit()
                st.error("🗑️ User deleted successfully!")

        cursor.close()
        conn.close()
    with chatbot:
    # Load API key
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        # Initialize GenAI client
        client = genai.Client(api_key=api_key)

        # Initialize session state
        if "mode" not in st.session_state:
            st.session_state.mode = "Admin"
        if "history" not in st.session_state:
            st.session_state.history = {"Normal": [], "Admin": []}

        st.subheader("💬 Assistant")
        
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            if st.button("🔄 Normal Mode", use_container_width=True):
                st.session_state.mode = "Normal"
                st.rerun()
        with mode_col2:
            if st.button("⚡ Admin Mode", use_container_width=True):
                st.session_state.mode = "Admin"
                st.rerun()
        
        st.write(f"**Current Mode:** {'🤖 General Chat' if st.session_state.mode == 'Normal' else '📊 Database Query'}")

        # Display chat history in a cleaner way
        st.markdown("---")
        st.write("**Chat History:**")
        
        for speaker, message in st.session_state.history[st.session_state.mode]:
            if speaker == "User":
                with st.chat_message("user"):
                    st.write(message)
            elif speaker == "AI":
                with st.chat_message("assistant"):
                    st.write(message)
            elif speaker == "AI (SQL Query)":
                with st.chat_message("assistant"):
                    st.code(message, language="sql")
            elif speaker == "Results":
                with st.chat_message("assistant"):
                    st.write("**Query Results:**")
                    st.dataframe(message)
            elif speaker == "Error":
                with st.chat_message("assistant"):
                    st.error(message)

        st.markdown("---")
        
        if user_input := st.chat_input("Type your message here..."):
            st.session_state.history[st.session_state.mode].append(("User", user_input))
            
            with st.chat_message("user"):
                st.write(user_input)
            
            # Handle different modes
            with st.chat_message("assistant"):
                if st.session_state.mode == "Admin":
                    handle_admin_query(user_input, client)
                else:
                    handle_normal_chat(user_input, client)


            # Clear chat button
            if st.session_state.history[st.session_state.mode]:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.history[st.session_state.mode] = []
                    st.rerun()

    # -----------------------------
    # Analytics Section
    # -----------------------------
    st.subheader("📈 User Analytics")

    # Get user data for analytics
    conn = db.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT name, email, age, sex, status, time_created FROM accounts")
    users_data = cursor.fetchall()
    cursor.close()
    conn.close()

    df_users = pd.DataFrame(users_data)

    if not df_users.empty:
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_users = len(df_users)
            st.metric("Total Users", total_users)
        
        with col2:
            active_users = len(df_users[df_users['status'] == 1])
            st.metric("Active Users", active_users)
        
        with col3:
            inactive_users = len(df_users[df_users['status'] == 0])
            st.metric("Inactive Users", inactive_users)
                # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.write("**Gender Distribution**")
            gender_counts = df_users['sex'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        
        with chart_col2:
            st.write("**Age Distribution**")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df_users['age'], bins=15, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Age')
            ax.set_ylabel('Number of Users')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Registration trends
        st.write("**Registration Trends**")
        df_users['time_created'] = pd.to_datetime(df_users['time_created'])
        df_users['registration_date'] = df_users['time_created'].dt.date
        daily_registrations = df_users.groupby('registration_date').size()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        daily_registrations.plot(ax=ax, marker='o', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Registrations')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        # User status breakdown
        st.write("**User Status Overview**")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            status_counts = df_users['status'].value_counts()
            status_counts.index = status_counts.index.map({0: 'Inactive', 1: 'Active'})
            st.dataframe(status_counts)
        
        with status_col2:
            st.write("**Recent Registrations**")
            recent_users = df_users.nlargest(5, 'time_created')[['name', 'email', 'time_created']]
            st.dataframe(recent_users)

        st.subheader("🖥️ System Status")

        sys_col1, sys_col2, sys_col3 = st.columns(3)

        with sys_col1:
            st.metric("Database Connection", "✅ Active")

        with sys_col2:
            # Count recent registrations (last 7 days)
            week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
            recent_count = len(df_users[df_users['time_created'] > week_ago])
            st.metric("New Users (7 days)", recent_count)

        with sys_col3:
            active_percentage = (active_users / total_users * 100) if total_users > 0 else 0
            st.metric("Activation Rate", f"{active_percentage:.1f}%")

    else:
        st.info("No user data available for analytics.")
            

    if st.sidebar.button("Logout"):
        st.session_state.auth = None
        st._set_query_params(page="login")
        st.rerun()

    st.sidebar.subheader("🚀 Quick Actions")

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    if st.sidebar.button("📥 Export Users"):
        csv = df_users.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="users_export.csv",
            mime="text/csv"
        )
# ---------------- User Dashboard ----------------
elif route == "user_ui" and st.session_state.auth == "user":
    #initializing database
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM accounts where email = \"{st.session_state.email}\"")
        results = cursor.fetchone()
        name = results[0]
        st.sidebar.image("Images/logo.png")
        st.title("👤 User Dashboard")
        st.write(f"Welcome User: **{name[0].upper() + name[1:] }**")
        st.write("This is your personal dashboard.")

        # -------------------------------
        # 1. Generate synthetic dataset
        # -------------------------------
        weathers = ["sunny", "rainy", "foggy"]
        roads = ["dry", "wet", "icy"]
        fatigue = ["yes", "no"]
        alcohol = ["yes", "no"]
        times = ["day", "night"]

        def decide_safe(weather, road, fatigue, alcohol, time):
            """Simple rule-based logic to label data"""
            if alcohol == "yes":
                return "no"
            if road == "icy":
                return "no"
            if fatigue == "yes" and time == "night":
                return "no"
            if np.random.rand() < 0.1:
                return "no"
            return "yes"

        def generate_dataset(N=500):
            records = []
            for _ in range(N):
                w = random.choice(weathers)
                r = random.choice(roads)
                f = random.choice(fatigue)
                a = random.choice(alcohol)
                t = random.choice(times)
                safe = decide_safe(w, r, f, a, t)
                records.append([w, r, f, a, t, safe])
            df = pd.DataFrame(records, columns=["Weather", "Road", "Fatigue", "Alcohol", "Time", "Safe"])
            return df

        # Generate dataset
        df = generate_dataset(1000)
        df_encoded = pd.get_dummies(df, columns=["Weather", "Road", "Fatigue", "Alcohol", "Time"])
        X = df_encoded.drop("Safe", axis=1)
        y = df_encoded["Safe"]
        le = LabelEncoder()
        y = le.fit_transform(df_encoded["Safe"])
        # Train Decision Tree
        # clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
        # Train/Test split (optional, but recommended)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = XGBClassifier(
        n_estimators=100,      # number of boosting rounds
        learning_rate=0.1,     # step size shrinkage
        max_depth=5,           # tree depth
        random_state=42,
        eval_metric="logloss"  # avoids warning
        )
        clf.fit(X_train, y_train)
        # 2. Streamlit UI
        # -------------------------------
        st.set_page_config(
            page_title="User Dashboard",
            page_icon="👤",
            layout="centered",  # "centered" or "wide"
            initial_sidebar_state="collapsed"  # "collapsed" or "expanded"
        )

        def assign_risk(row):
            risk_score = 0
            risk_factors = []
            
            # ---------------- Driver-related risk (More nuanced scoring) ----------------
            # Age risk: U-shaped curve (young and elderly are higher risk)
            if row['Driver_Age'] < 22:
                risk_score += 2
                risk_factors.append("Young driver (<22)")
            elif row['Driver_Age'] < 25:
                risk_score += 1
                risk_factors.append("Inexperienced driver (22-24)")
            elif row['Driver_Age'] > 75:
                risk_score += 2
                risk_factors.append("Elderly driver (>75)")
            elif row['Driver_Age'] > 65:
                risk_score += 1
                risk_factors.append("Senior driver (65-75)")
            
            # Experience interaction with age
            if row['Driver_Experience'] < 2:
                risk_score += 2
                risk_factors.append("Very inexperienced (<2 years)")
            elif row['Driver_Experience'] < 5:
                risk_score += 1
                risk_factors.append("Limited experience (2-5 years)")
            
            # Fatigue with progressive scoring
            if row['Fatigue_Level'] == "High":
                risk_score += 3
                risk_factors.append("High fatigue")
            elif row['Fatigue_Level'] == "Medium":
                risk_score += 1
                risk_factors.append("Medium fatigue")
            
            # Awake time with progressive risk
            if row['Driver_Awake_Time'] > 18:
                risk_score += 3
                risk_factors.append("Extreme awake time (>18h)")
            elif row['Driver_Awake_Time'] > 16:
                risk_score += 2
                risk_factors.append("Long awake time (16-18h)")
            elif row['Driver_Awake_Time'] > 14:
                risk_score += 1
                risk_factors.append("Extended awake time (14-16h)")
            
            # Speeding - major risk factor
            if row['Speeding'] == "Yes":
                risk_score += 3
                risk_factors.append("Speeding")
            
            # Vehicle maintenance
            if row['Last_Service_Months_Ago'] > 24:
                risk_score += 2
                risk_factors.append("Poor maintenance (>2 years)")
            elif row['Last_Service_Months_Ago'] > 12:
                risk_score += 1
                risk_factors.append("Overdue service (1-2 years)")
            
            # ---------------- Environmental Risk (Weighted by severity) ----------------
            # Visibility - critical factor
            if row['Visibility'] == "Poor":
                risk_score += 3
                risk_factors.append("Poor visibility")
            elif row['Visibility'] == "Moderate":
                risk_score += 1
                risk_factors.append("Moderate visibility")
            
            # Road surface conditions
            if row['Road_Surface_Conditions'] == "Icy":
                risk_score += 3
                risk_factors.append("Icy roads")
            elif row['Road_Surface_Conditions'] == "Wet":
                risk_score += 2
                risk_factors.append("Wet roads")
            
            # Weather conditions
            if row['Weather'] in ["Snowy", "Foggy"]:
                risk_score += 2
                risk_factors.append(f"{row['Weather']} weather")
            elif row['Weather'] == "Rainy":
                risk_score += 1
                risk_factors.append("Rainy weather")
            
            # Light conditions
            if row['Light_Conditions'] == "Night":
                risk_score += 2
                risk_factors.append("Night driving")
            elif row['Light_Conditions'] == "Dusk/Dawn":
                risk_score += 1
                risk_factors.append("Low light conditions")
            
            # Traffic density with time interaction
            if row['Traffic_Density'] > 0.8:
                risk_score += 2
                risk_factors.append("Heavy traffic")
            elif row['Traffic_Density'] > 0.6:
                risk_score += 1
                risk_factors.append("Moderate-heavy traffic")
            
            # Road type risk
            if row['Road_Type'] == "Mountainous":
                risk_score += 2
                risk_factors.append("Mountainous roads")
            elif row['Road_Type'] == "Rural":
                risk_score += 1
                risk_factors.append("Rural roads")
            
            # Temperature extremes
            if row['Temperature'] < -5 or row['Temperature'] > 35:
                risk_score += 1
                risk_factors.append("Extreme temperature")
            
            # ---------------- Risk Combinations (Multiplicative effects) ----------------
            # Dangerous combinations
            dangerous_combinations = [
                (row['Fatigue_Level'] == "High" and row['Speeding'] == "Yes", 2, "Fatigue + Speeding"),
                (row['Visibility'] == "Poor" and row['Road_Surface_Conditions'] in ["Icy", "Wet"], 2, "Poor visibility + Bad road conditions"),
                (row['Light_Conditions'] == "Night" and row['Driver_Age'] < 25, 2, "Young driver at night"),
                (row['Traffic_Density'] > 0.7 and row['Weather'] != "Clear", 1, "Heavy traffic + Bad weather")
            ]
            
            for condition, points, description in dangerous_combinations:
                if condition:
                    risk_score += points
                    risk_factors.append(description)
            
            # ---------------- Final Risk Assessment ----------------
            # More graduated risk levels based on real-world data
            if risk_score >= 12:
                return 1  # High risk
            elif risk_score >= 8:
                return 1  # Medium-high risk
            elif risk_score >= 5:
                return 1  # Medium risk
            else:
                return 0  # Low risk

        st.set_page_config(page_title="Driving Risk Predictor", page_icon="🚗", layout="centered")
        st.write("Enter all required values to predict the probability of driving risk.")

        # -----------------------------
        # 1. Synthetic dataset generation
        # -----------------------------
        def generate_synthetic_data(n=2000, seed=42):
            np.random.seed(seed)
            data = []
            for _ in range(n):
                age = np.random.randint(18, 80)
                exp = max(0, age - np.random.randint(16, 25))  # exp grows with age
                awake = np.random.randint(1, 20)
                fatigue = np.random.choice(["Low", "Medium", "High"])
                speeding = np.random.choice(["Yes", "No"])
                service = np.random.randint(0, 60)
                visibility = np.random.choice(["Good", "Moderate", "Poor"])
                light = np.random.choice(["Daylight", "Night", "Dusk/Dawn"])
                road_surface = np.random.choice(["Dry", "Wet", "Icy"])
                weather = np.random.choice(["Clear", "Rainy", "Foggy", "Snowy"])
                road_type = np.random.choice(["Highway", "Urban", "Rural"])
                landscape = np.random.choice(["Flat", "Hilly", "Mountainous"])
                traffic_density = np.round(np.random.uniform(0, 1), 2)
                temp = np.random.randint(-10, 45)
                time_day = np.random.choice(["Morning", "Afternoon", "Evening", "Night"])

                data.append([
                    age, exp, awake, fatigue, speeding, service,
                    visibility, light, road_surface, weather, road_type, landscape,
                    traffic_density, temp, time_day, 
                ])

            df = pd.DataFrame(data, columns=[
                "Driver_Age","Driver_Experience","Driver_Awake_Time","Fatigue_Level","Speeding",
                "Last_Service_Months_Ago",
                "Visibility","Light_Conditions","Road_Surface_Conditions","Weather","Road_Type","Landscape",
                "Traffic_Density","Temperature","Time_of_Day"
            ])
            df["Risk"] = df.apply(assign_risk, axis=1)
            return df

        df = generate_synthetic_data()

        # Encode categorical variables
        df_encoded = df.copy()
        for col in ["Fatigue_Level","Speeding","Visibility","Light_Conditions",
                    "Road_Surface_Conditions","Weather","Road_Type","Landscape","Time_of_Day"]:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

        X = df_encoded.drop("Risk", axis=1)
        y = df_encoded["Risk"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier(eval_metric="logloss")
        model.fit(X_train, y_train)

        # -----------------------------
        # 2. Show model performance
        # -----------------------------
        st.subheader("📊 Model Performance on Synthetic Test Data")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"**Accuracy:** {acc:.3f}")
        st.write(f"**ROC AUC:** {auc:.3f}")

        cm_df = pd.DataFrame(cm, index=["Actual Safe (0)", "Actual Risk (1)"],
                                columns=["Predicted Safe (0)", "Predicted Risk (1)"])
        st.dataframe(cm_df)

        # -----------------------------
        # 3. Streamlit user input form
        # -----------------------------
        st.subheader("🔧 Input Driving Factors")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Driver Age", 18, 100, 30)
            exp = st.number_input("Driver Experience (years)", 0, 80, 5)
            awake = st.number_input("Driver Awake Time (hours)", 0, 24, 8)
            fatigue = st.selectbox("Fatigue Level", ["Low", "Medium", "High"])
            speeding = st.selectbox("Speeding", ["Yes", "No"])
            service = st.number_input("Last Service (months ago)", 0, 120, 6)

       
            visibility = st.selectbox("Visibility", ["Good", "Moderate", "Poor"])
            # Weather selection
            weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy"])
        with col2:
            # Road surface depends on weather
            if weather == "Clear":
                road_surface_options = ["Dry"]
            elif weather == "Rainy":
                road_surface_options = ["Wet"]
            elif weather == "Snowy":
                road_surface_options = ["Icy", "Wet"]
            elif weather == "Foggy":
                road_surface_options = ["Dry", "Wet"]
            else:
                road_surface_options = ["Dry", "Wet", "Icy"]

            road_surface = st.selectbox("Road Surface", road_surface_options)

            # Time of day selection
            time_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

            # Light conditions depend on time of day
            if time_day in ["Morning", "Afternoon"]:
                light_options = ["Daylight", "Dusk/Dawn"]
            elif time_day == "Evening":
                light_options = ["Dusk/Dawn", "Night"]
            elif time_day == "Night":
                light_options = ["Night"]

            light = st.selectbox("Light Conditions", light_options)

            road_type = st.selectbox("Road Type", ["Highway", "Urban", "Rural"])
            landscape = st.selectbox("Landscape", ["Flat", "Hilly", "Mountainous"])
            traffic_density = st.slider("Traffic Density", 0.0, 1.0, 0.5)
            temp = st.number_input("Temperature (°C)", -20, 50, 25)

        # -----------------------------
        # 4. Encode user input
        # -----------------------------
        input_dict = {
            "Driver_Age": age,
            "Driver_Experience": exp,
            "Driver_Awake_Time": awake,
            "Fatigue_Level": fatigue,
            "Speeding": speeding,
            "Last_Service_Months_Ago": service,
            "Visibility": visibility,
            "Light_Conditions": light,
            "Road_Surface_Conditions": road_surface,
            "Weather": weather,
            "Road_Type": road_type,
            "Landscape": landscape,
            "Traffic_Density": traffic_density,
            "Temperature": temp,
            "Time_of_Day": time_day
        }

        input_df = pd.DataFrame([input_dict])

        # Encode with same method
        for col in ["Fatigue_Level","Speeding","Visibility","Light_Conditions",
                    "Road_Surface_Conditions","Weather","Road_Type","Landscape","Time_of_Day"]:
            le = LabelEncoder()
            le.fit(df[col])  # fit on training categories
            input_df[col] = le.transform(input_df[col])

        # -----------------------------
        # 5. Prediction
        # -----------------------------
        if st.button("Predict Risk"):
            prob = model.predict_proba(input_df)[0][1]
            st.metric("Predicted Probability of Risk", f"{prob*100:.2f}%")
            if prob > 0.7:
                st.error("⚠️ High Risk! Drive carefully.")
            elif prob > 0.4:
                st.warning("⚠️ Medium Risk. Stay alert.")
            else:
                st.success("✅ Low Risk. Safe driving conditions.")

        with st.expander("See feature importance"):
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_importance(model, ax=ax, importance_type="gain", show_values=False)
            st.pyplot(fig)

        with st.expander("See Confusion Matrix"):
            st.subheader("🔢 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Safe (0)", "Risk (1)"], yticklabels=["Safe (0)", "Risk (1)"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
         
        if st.button("Get geolocation"):
            query_ip, city, lat, lon = fetch_geolocation()
            if city:
                st.write(f"**City:** {city}")
                st.write(f"**Latitude:** {lat}")
                st.write(f"**Longitude:** {lon}")

                # Show on map
                df = pd.DataFrame([[lat, lon]], columns=["lat", "lon"])
                st.map(df, zoom=14, use_container_width=True)
            else:
                st.error("❌ Could not retrieve your geolocation data.")
        with st.sidebar:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")

            st.markdown("---")
            st.subheader("💬 Quick Assistant")
            
            # Initialize sidebar chat
            if "sidebar_chat_history" not in st.session_state:
                st.session_state.sidebar_chat_history = []
            
            # Display last 3 messages in sidebar
            for speaker, message in list(st.session_state.sidebar_chat_history)[-3:]:
                if speaker == "User":
                    st.write(f"**You:** {message}")
                    st.write("---")
                else:
                    st.write(f"**Assistant:** {message}")
                    st.write("---")
            
            # Simple input in sidebar
            sidebar_input = st.sidebar.text_input("Ask about driving safety...", key="sidebar_input")
            submit = st.sidebar.button("Send")
            if submit and api_key:
                st.session_state.sidebar_chat_history.append(("User", sidebar_input))
                
                try:
                    client = genai.Client(api_key=api_key)
                    response = handle_user_chat_query(sidebar_input, client)
                    st.session_state.sidebar_chat_history.append(("AI", response))
                    st.rerun()
                    sidebar_input = ""
                except Exception as e:
                    st.error("Failed to get response")

                # Clear chat button
        if st.session_state.sidebar_chat_history:
            if st.sidebar.button("🗑️ Clear Chat", key="user_clear_chat"):
                st.session_state.sidebar_chat_history = []
                st.rerun()
                st.sidebar.write("")
        if st.sidebar.button("Logout"):
            st.session_state.auth = None
            st._set_query_params(page="login")
            st.rerun()
                
elif route == "sign_up" and st.session_state.auth == "signup":
    st.set_page_config(
        page_title="Sign up",
        page_icon="📃",
        layout="centered",  # "centered" or "wide"
        initial_sidebar_state="collapsed"  # "collapsed" or "expanded"
    )
    st.image("Images/logo.png")
    st.title("Sign up")
    with st.form("signup_form"):
        firstcol, seccol = st.columns(2)
        with firstcol:

            name = st.text_input("Full Name")
            email = st.text_input("Email")
            age = st.number_input("Age", min_value=18, max_value=100, step=1)
            sex = st.selectbox("Sex", ["Male", "Female"])
        with seccol:
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            account = st.selectbox("Sign up as",["User", "Admin"])
            spacing, col1,col2 = st.columns([1.7,1,1])
            submit = col1.form_submit_button("Sign Up")
            back = col2.form_submit_button("Back")

    if submit:
        if account == "User":
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif not name or not email or not password:
                st.error("All fields are required!")
            else:
                success, message = db.register_user(name, email, age, sex, password,account)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        elif account == "Admin":
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif not name or not email or not password:
                st.error("All fields are required!")
            else:
                success, message = db.register_admin(name, email, age, sex, password,account)
                if success:
                    st.success(message)
                
                else:
                    st.error(message)
                    st.rerun()
        
    if back:
        st.session_state.auth = None
        st._set_query_params(page="login")
        st.rerun()

# ---------------- Unauthorized Access ----------------
else:
    st.warning("⚠️ Unauthorized. Please login.")
    st._set_query_params(page="login")
