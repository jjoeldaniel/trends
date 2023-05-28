use std::fs;
use rust_bert::{pipelines::keywords_extraction::{KeywordExtractionModel, Keyword}, RustBertError};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rust_stemmers::{Algorithm, Stemmer};


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

    let _en_stemmer = Stemmer::create(Algorithm::English);

    let json_contents = fs::read_to_string(file_path).expect("Error reading file").to_lowercase();

    // TODO: Stem messages

    let message_data: MessageData = serde_json::from_str(&json_contents).expect("Error deserializing JSON");
    let mut inputs: Vec<String> = Vec::new();

    if let Some(messages) = message_data.messages.get(..) {
        for message in messages {
            
            // clear empty
            if !message.text.is_empty() {
                inputs.push(String::from(&message.text));
            }

        }  
    }
    inputs
}

fn extract_keywords(inputs: Vec<String>) -> Result<Vec<Vec<Keyword>>, RustBertError> {
    let keyword_extraction_model = KeywordExtractionModel::new(Default::default())?;
    keyword_extraction_model.predict(&[inputs.join("\n")])
}

fn main() {
    let messages = read_input("./data/data.json");
    let keywords = extract_keywords(messages);

    if let Ok(n) = keywords {
        for m in n {
            for k in m {
                println!("{:?}\n", k);
            }
        } 
    }
}

