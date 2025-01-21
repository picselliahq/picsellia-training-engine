import json
import os


class SessionManager:
    SESSION_FILE = os.path.join("cli", ".session.json")

    def __init__(self):
        self.session: dict = {"global": {}, "processings": {}}
        self.load_session()

    def load_session(self):
        """Load session data from the JSON file, or initialize if it doesn't exist."""
        if os.path.exists(self.SESSION_FILE):
            with open(self.SESSION_FILE, "r") as file:
                self.session = json.load(file)
        else:
            # Ensure default structure if session file is missing
            self.session = {"global": {}, "processings": {}}

    def save_session(self):
        """Save session data to the JSON file."""
        with open(self.SESSION_FILE, "w") as file:
            json.dump(self.session, file, indent=4)

    def ensure_session_initialized(self):
        """Ensure global session is initialized, or prompt the user to initialize."""
        if not self.session["global"]:
            self.initialize_global()

    def initialize_global(self):
        """Prompt user to initialize global session data."""
        print(
            "Global session is not initialized. Please provide the following details:"
        )
        api_token = input("API token: ")
        organization_id = input("Organization ID: ")
        self.session["global"] = {
            "api_token": api_token,
            "organization_id": organization_id,
        }
        self.save_session()

    def ensure_processing_exists(self, processing_name):
        """Ensure a processing exists, or guide the user to add it."""
        if processing_name not in self.session.get("processings", {}):
            print(f"Processing '{processing_name}' does not exist.")
            print("Please add it using the `add-processing` command.")
            return False
        return True

    def add_processing(self, name, data):
        """Add or update a processing configuration."""
        self.session["processings"][name] = data
        self.save_session()

    def remove_processing(self, name):
        """Remove a processing configuration by name."""
        if name in self.session["processings"]:
            del self.session["processings"][name]
            self.save_session()

    def list_processings(self):
        """List all registered processing names."""
        return list(self.session.get("processings", {}).keys())

    def get_processing(self, name):
        """Retrieve a processing configuration by name."""
        return self.session["processings"].get(name)

    def get_global(self):
        """Retrieve global session data."""
        return self.session["global"]


session_manager = SessionManager()
