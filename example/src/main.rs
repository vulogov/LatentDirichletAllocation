extern crate log;

use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::*;

//
// This is a text corpus. Taken from my essay published on LinkledIn.
//
const DOCUMENTS: &[&str] = &[
    "In the realm of AI, particularly with the rise of large language models and modern machine learning, it's essential to reflect on the value of classic expert systems from the 1970s to the 1990s",
    "While contemporary AI often operates as a black box, classic expert systems operate on a white box principle, in which the reasoning process is transparent: IF these conditions are true, THEN this conclusion",
    "This clarity is crucial in fields like observability, where understanding the decision-making chain is paramount",
    "Expert systems are built on human knowledge, excelling in areas where data are limited or nonexistent, yet experts possess deep theoretical and experiential insights. In contrast, modern AI typically requires large amounts of data, which may not come easily in telemetry collection applications",
    "Predictability is another strength of expert systems. They consistently produce the same output for identical inputs, making them reliable",
    "This predictability is beneficial in observability, whereas modern AI can exhibit statistical variability, leading to different answers and unpredictable behavior with slight shifts in input data",
];

fn main() {
    let mut lda = latentdirichletallocation::default(DOCUMENTS);
    let iterations = 800;
    println!("Training LDA (K={}, iters={iterations})...", &lda.k);
    lda.train(iterations);
    let topics = lda.topics(8);
    let mut topics_table = Table::new();
    topics_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .add_row(vec![
            Cell::new("Topic ID").fg(Color::Red), Cell::new("Words").fg(Color::White),
        ]);
    for t in topics.keys() {
        let row = topics.get(t).unwrap();
        let mut words: String = "".to_owned();
        for w in row.keys() {
            let score = row.get(w).unwrap();
            words.push_str(&format!("{}[{}] ", &w, score));
        }
        topics_table.add_row(vec![
            Cell::new(format!("{}", &t)),
            Cell::new(words)
        ]);
    }
    println!("{}", topics_table);
}
