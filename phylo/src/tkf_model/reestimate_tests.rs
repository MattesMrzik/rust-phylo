use crate::tkf_model::{
    possibilities_for_choices, };

#[test]
fn tkf_reestimate_possibilities_for_choices_no_choice() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![];
    let possibilities = possibilities_for_choices(&choices, &no_choices);
    let expected = vec![[true, false, false, true, false]];
    assert_eq!(possibilities, expected);
}

#[test]
fn tkf_reestimate_possibilities_for_choices_one() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![0];
    let mut possibilities = possibilities_for_choices(&choices, &no_choices);
    let mut expected = vec![
        [true, false, false, true, false],
        [false, false, false, true, false],
    ];
    // sort
    possibilities.sort();
    expected.sort();
    assert_eq!(possibilities, expected);
}

#[test]
fn tkf_reestimate_possibilities_for_choices() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![1, 3];
    let mut possibilities = possibilities_for_choices(&choices, &no_choices);
    let mut expected = vec![
        [true, false, false, false, false],
        [true, false, false, true, false],
        [true, true, false, false, false],
        [true, true, false, true, false],
    ];
    // sort
    possibilities.sort();
    expected.sort();
    assert_eq!(possibilities, expected);
}

#[test]
fn tkf_reestimate_possibilities_for_choices_all() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![0, 1, 2, 3, 4];
    let possibilities = possibilities_for_choices(&choices, &no_choices);
    let expected = (0..32)
        .map(|i| {
            [
                (i & 0b00001) != 0,
                (i & 0b00010) != 0,
                (i & 0b00100) != 0,
                (i & 0b01000) != 0,
                (i & 0b10000) != 0,
            ]
        })
        .collect::<Vec<[bool; 5]>>();

    assert_eq!(possibilities, expected);
}
