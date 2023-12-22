import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def normalize(dict):
    result = {}
    total = sum(dict.values())
    for key in dict:
        result[key] = (dict[key] / total) * 100
    return result

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result = {}
    corpus_total_pages_count = len(corpus)
    pages = corpus[page]
    random_transition_probabilities = 1 - damping_factor
    random_transition_probability = random_transition_probabilities / corpus_total_pages_count
    current_page_links = len(pages)
    free_probabilities = 1 - random_transition_probabilities
    rank_per_page = free_probabilities / current_page_links if current_page_links > 0 else 0

    for name in corpus:
        rank = random_transition_probability
        if name in pages:
            rank += rank_per_page
        result[name] = rank

    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    index = random.randrange(0, len(corpus))
    pages = list(corpus.keys())
    page = pages[index]
    samples = {page: 0 for page in pages}

    for i in range(n):
        result = transition_model(corpus, page, damping_factor)
        rnd = random.uniform(0, 1)
        for key in result:
            rnd -= result[key]
            if rnd <= 0:
                page = key
                samples[key] += 1
                break

    return normalize(samples)


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    precision = 0.001
    pages = corpus.keys()
    pages_count = len(pages)
    initial = 1 / pages_count
    ranks = {page: initial for page in corpus}
    inbound_links = {page: set() for page in corpus}
    for page in corpus:
        for link in corpus[page]:
            if link in inbound_links:
                inbound_links[link].add(page)

    k = (1 - damping_factor) / pages_count

    while True:
        changed = False
        ranks_update = {}
        for page in pages:
            acc = 0
            for linking_page in inbound_links[page]:
                acc += ranks[linking_page] / len(corpus[linking_page])
            update = k + damping_factor * acc
            prev = ranks[page]
            diff = prev - update
            if abs(diff) > precision:
                ranks_update[page] = update
                changed = True
        if not changed:
            break
        else:
            ranks.update(ranks_update)

    return normalize(ranks)


if __name__ == "__main__":
    main()
