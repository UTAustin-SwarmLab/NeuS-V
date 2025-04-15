import datetime
import json
import os


class LLM:
    def __init__(self, client, save_dir="outputs"):
        self.client = client
        self.history = []
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def prompt(self, p):
        # Add user message to history
        user_message = {"role": "user", "content": [{"type": "text", "text": p}]}
        self.history.append(user_message)

        # Create messages list with history
        messages = self.history.copy()

        response = self.client.chat.completions.create(
            model="o1-mini-2024-09-12",
            messages=self.history,
            store=False,
        )

        # Extract assistant response
        assistant_response = response.choices[0].message.content

        # Add assistant response to history
        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        self.history.append(assistant_message)

        return assistant_response

    def save_history(self, filename="conversation_history.json"):
        """Save conversation history to a JSON file"""
        if not self.save_dir:
            return

        # Add timestamp to filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, extension = os.path.splitext(filename)
        timestamped_filename = f"{base_name}_{timestamp}{extension}"

        save_path = os.path.join(self.save_dir, timestamped_filename)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)
            print(f"Conversation history saved to: {save_path}")
        except Exception as e:
            print(f"Failed to save conversation history: {e}")

    def __del__(self):
        """Destructor: Saves conversation history when the object is destroyed"""
        if self.save_dir:
            self.save_history()
