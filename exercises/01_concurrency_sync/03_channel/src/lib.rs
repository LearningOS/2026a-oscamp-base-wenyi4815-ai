//! # Channel Communication
//!
//! In this exercise, you will use `std::sync::mpsc` channels to pass messages between threads.
//!
//! ## Concepts
//! - `mpsc::channel()` creates a multiple producer, single consumer channel
//! - `Sender::send()` sends a message
//! - `Receiver::recv()` receives a message
//! - Multiple producers can be created via `Sender::clone()`

use std::sync::mpsc;
use std::thread;

/// Create a producer thread that sends each element from items into the channel.
/// The main thread receives all messages and returns them.
pub fn simple_send_recv(items: Vec<String>) -> Vec<String> {
    // 1. Create channel, split into sender and receiver
    let (tx, rx) = mpsc::channel();

    // 2. Spawn producer thread, move tx and items into closure
    thread::spawn(move || {
        for item in items {
            tx.send(item).expect("Failed to send message");
        }
        // tx is automatically dropped when thread closure ends
    });

    // 3. Main thread collect all received messages
    let mut collected = Vec::new();
    // recv() returns Err when all senders are dropped
    while let Ok(msg) = rx.recv() {
        collected.push(msg);
    }

    collected
}

/// Create `n_producers` producer threads, each sending a message in format `"msg from {id}"`.
/// Collect all messages, sort them lexicographically, and return.
///
/// Hint: Use `tx.clone()` to create multiple senders. Note that the original tx must also be dropped.
pub fn multi_producer(n_producers: usize) -> Vec<String> {
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();

    // Spawn one thread per producer id
    for id in 0..n_producers {
        let tx_clone = tx.clone(); // clone sender for each thread
        let handle = thread::spawn(move || {
            let msg = format!("msg from {}", id);
            tx_clone.send(msg).expect("Send failed");
            // cloned sender drops here
        });
        handles.push(handle);
    }

    // Drop original tx immediately! Otherwise rx.recv() will block forever
    drop(tx);

    // Wait all threads finish sending (optional but clean practice)
    for h in handles {
        h.join().expect("Producer thread panicked");
    }

    // Collect all messages
    let mut messages = Vec::new();
    while let Ok(m) = rx.recv() {
        messages.push(m);
    }

    // Sort lexicographically as required
    messages.sort();
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_send_recv() {
        let items = vec!["hello".into(), "world".into(), "rust".into()];
        let result = simple_send_recv(items.clone());
        assert_eq!(result, items);
    }

    #[test]
    fn test_simple_empty() {
        let result = simple_send_recv(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_multi_producer() {
        let result = multi_producer(3);
        assert_eq!(
            result,
            vec![
                "msg from 0".to_string(),
                "msg from 1".to_string(),
                "msg from 2".to_string(),
            ]
        );
    }

    #[test]
    fn test_multi_producer_single() {
        let result = multi_producer(1);
        assert_eq!(result, vec!["msg from 0".to_string()]);
    }
}