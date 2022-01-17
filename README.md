# WordleCracker
## Information theory-based Wordle solver

Core algorithm runs as follows:
 - Choose a guess word.
 - Associate each possible scoring outcome (e.g. grey, green, yellow, gray, gray) with a separate ternary encoding. Compile across all possible target words.
 - Count the total number of targets that would have given the same scoring outcome by counting up the number of associated ternary encodings. Convert to probabilities by dividing by the total length of the list of potential targets.
 - Calculate the entropy of the guess as -sum(Pi * log_2(Pi)), summing over all of the encodings.
 - Repeat for all other guess words. The guess with the largest associated entropy cuts the search space down by the greatest amount, and so is selected as the best candidate.

Repeated application of this algorithm (O(n^2)), along with elimination of potential targets that are inconsistent with the resulting scoreing outcome, allows the search space to be whittled down to ~5-10 candidates within 2 guesses. Correct selection from within this candidate list is often more luck than judgement.

The best starting word is predicted to be 'tares', with an entropy of ~6 bits. This cuts the initial search space (~12,000) down by a factor of ~64x, on average.
