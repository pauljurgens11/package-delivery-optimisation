# Graph Generator

- `graph_generator.py`
  - used to generate various kinds of graphs
- `test_cases.py`
  - specifies which graphs we want to generate to hit different edge cases for our algorithm analysis
  - the specific test methodology and why we use these test cases is described in the root README
- `visualize_graph.py`
  - visualize a graph to ensure generation actually works.
- `generated_graphs/`
  - contains the graphs we generated that will be used in tests
  - naming convention:
    - the letter at the start indicates test case (A to F) desctibed in the root README
    - what follows is a few word description of the test
    - n1000 to n100000 is the number of vertexes
    - seed0 to seed2 describes randomness

## Usage

To generate graphs, edit the main method in `graph_generator.py`. It automatically imports the various test cases from `test_cases.py`.

Usage examples of `visualize_graph.py` are found in the file as a top level comment.
