import streamlit_authenticator as stauth

# User details
new_username = "user_name"
new_password = "user_password"  # This is the password you want to hash
new_name = "user_name"
new_email = "user_email"



# Initialize the Hasher
hashed_passwords = stauth.Hasher.hash(new_password)
# hashed_passwords = hasher.generate()

# Load current config from YAML
import yaml
import os

# --- Load Config for Authentication ---
config_path = os.getenv("USER_CONFIG_PATH", "../DATA/users_config.yaml")
with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

if config['credentials']['usernames'] is None:
    config['credentials']['usernames'] = dict()

# Add new user to the credentials section
config['credentials']['usernames'][new_username] = {
    'name': new_name,
    'email': new_email,
    'username': new_username,
    'password': hashed_passwords  # Use the hashed password
}

# Save updated config back to YAML file
with open(config_path, 'w') as file:
    yaml.dump(config, file)

# Print success message (or show it in Streamlit)
print("New user added successfully!")
