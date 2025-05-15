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


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    N = len(corpus)
    prob = dict()
    links = corpus[page]
    if links:
        for p in corpus:
            prob[p] = (1 - damping_factor) / N
        for link in links:
            prob[link] += damping_factor / len(links)
    else:
        for p in corpus:
            prob[p] = 1 / N

    return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {page: 0 for page in corpus}
    pages = list(corpus.keys())
    page = random.choice(pages)
    for i in range(n):
        counts[page] += 1
        probs = transition_model(corpus, page, damping_factor)
        page = random.choices(
            population=list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]

    pagerank = {page: count / n for page, count in counts.items()}
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}
    converge = False
    for page in corpus:
        if not corpus[page]:
            corpus[page] = set(corpus.keys())

    while not converge:
        new_ranks = dict()
        converge = True
        for page in corpus:
            total = 0
            for i in corpus:
                if page in corpus[i]:
                    total += pagerank[i] / len(corpus[i])
            new_rank = (1 - damping_factor) / N+damping_factor*total

            if abs(new_rank-pagerank[page]) > 0.001:
                converge = False
            new_ranks[page] = new_rank
        pagerank = new_ranks.copy()

    return pagerank


if __name__ == "__main__":
    main()
