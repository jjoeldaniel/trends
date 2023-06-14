use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

#[derive(Serialize, Deserialize)]
struct MessageData {
    channel_id: String,
    channel_name: String,
    authors: HashMap<String, Author>, // author_id -> Author
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize)]
struct Author {
    name: String,
    avatar: String,
}

#[derive(Serialize, Deserialize)]
struct Message {
    timestamp: String,
    author_id: String,
    text: String,
}

fn read_input(file_path: &str) -> Vec<String> {
    let json_contents = fs::read_to_string(file_path)
        .expect("Error reading file")
        .to_lowercase();

    let message_data: MessageData =
        serde_json::from_str(&json_contents).expect("Error deserializing JSON");
    let mut inputs: Vec<String> = Vec::new();

    if let Some(messages) = message_data.messages.get(..) {
        for message in messages {
            // clear empty
            if !message.text.is_empty() {
                let x =
                    message_data.authors[&message.author_id].name.clone() + ": " + &message.text;
                inputs.push(x);
            }
        }
    }
    inputs
}

fn main() {
    let messages = read_input("./data/data.json");

    // Output to ./data/output.txt
    fs::write("./data/output.txt", messages.join("\n")).expect("Unable to write file");
}
