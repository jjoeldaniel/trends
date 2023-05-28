use std::fs;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize)]
struct MessageData {
    channel_id: String,
    channel_name: String,
    authors: HashMap<String, Author>,
    messages: Vec<Message>
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

fn read_input(file_path: &str) {

    let json_contents = fs::read_to_string(file_path).expect("Error reading file");
    let message_data: MessageData = serde_json::from_str(&json_contents).expect("Error deserializing JSON");

    if let Some(messages) = message_data.messages.get(..5) {
        for message in messages {
            println!("Timestamp: {}", message.timestamp);
            println!("Author ID: {}", message.author_id);
            println!("Text: {}\n", message.text);
        }  
    }
}

fn main() {
    read_input("./data/test_data2.json");
}

