use std::fs;
use rust_bert::{pipelines::keywords_extraction::{KeywordExtractionModel, Keyword}, RustBertError};
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

fn read_input(file_path: &str) -> Vec<String> {

    let json_contents = fs::read_to_string(file_path).expect("Error reading file");
    let message_data: MessageData = serde_json::from_str(&json_contents).expect("Error deserializing JSON");
    let mut inputs: Vec<String> = Vec::new();

    if let Some(messages) = message_data.messages.get(..) {
        for message in messages {
            
            // clear empty
            if message.text != "" {
                inputs.push(String::from(&message.text));
            }

        }  
    }
    return inputs;
}

fn extract_keywords(inputs: Vec<String>) -> Result<Vec<Vec<Keyword>>, RustBertError> {
    let keyword_extraction_model = KeywordExtractionModel::new(Default::default())?;
    return Ok(keyword_extraction_model.predict(&[inputs.join("\n")])?);
}

fn main() {
    let messages = read_input("./data/test_data2.json");
    let keywords = extract_keywords(messages);

    if let Ok(n) = keywords {
        for m in n {
            for k in m {
                println!("{:?}", k);
            }
        } 
    }
}

