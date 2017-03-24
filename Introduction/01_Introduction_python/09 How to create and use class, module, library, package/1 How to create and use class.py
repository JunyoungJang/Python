# -*- coding: utf8 -*-

# http://cs231n.github.io/python-numpy-tutorial/#python-basic
#
# Classes
# The syntax for defining classes in Python is straightforward:

#'''
print 'Construction of class using construct, instance, and destructor ---'
class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable
        self._city = 'Seoul'
    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name
    # Destructor
    def __del__(self):
        print "Objects generated using class Greeter destructed!"
g = Greeter('Fred')  # Construct an instance of the Greeter class
g.name = 'Paul'
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
print g._city
del g
#'''
'''
'''
#'''
#'''
'''
print 'class HousePark ---'
class HousePark:
    last_name = "박"
    def __init__(self, first_name):
        self.full_name = self.last_name + first_name
    def __del__(self):
        print "Objects generated using class HousePark destructed!"
    def __add__(self, other):
        print "%s, %s 결혼했네" % (self.full_name, other.full_name)
    def __sub__(self, other):
        print "%s, %s 이혼했네" % (self.full_name, other.full_name)
    def fight(self, other):
        print "%s, %s 싸우네" % (self.full_name, other.full_name)
    def love(self, other):
        print "%s, %s 사랑에 빠졌네" % (self.full_name, other.full_name)
    def travel(self, where):
        print "%s, %s여행을 가다." % (self.full_name, where)
a = HousePark('응용')
a.travel('부산') # 박응용, 부산여행을 가다.
del a           # Objects generated using class HousePark destructed!
print 'class HouseKim inherited from class HousePark with overriding ---'
class HouseKim(HousePark):                                         # inheritance
    last_name = '김'                                                # overriding
    def __del__(self):                                             # overriding
        print "Objects generated using class HouseKim destructed!" # overriding
    def travel(self, where, day):                                  # overriding
        print "%s, %s여행 %d일 가네." % (self.full_name, where, day)  # overriding
pey = HousePark("응용")
juliet = HouseKim("줄리엣")
pey.travel("부산")       # 박응용, 부산여행을 가다.
juliet.travel("부산", 3) # 김줄리엣, 부산여행 3일 가네.
pey.love(juliet)        # 박응용, 김줄리엣 사랑에 빠졌네
pey + juliet            # 박응용, 김줄리엣 결혼했네
pey.fight(juliet)       # 박응용, 김줄리엣 싸우네
pey - juliet            # 박응용, 김줄리엣 이혼했네
                        # Objects generated using class HousePark destructed!
                        # Objects generated using class HouseKim destructed!
'''
'''
'''
#'''
#'''
'''
# NumPy is a numerical library in Python
# - Provides matrices and tools to manipulate them
# - Plus a large library of linear algebra operations

# What criteria can we use to recommend papers?
# 1. The way the paper was rated by other people
# 2. The similarity between those raters and the previous ratings for an individual

# Plan
# 1. Process people's ratings of various papers and store in NumPy array
# 2. Introduce two similarity measures
# 3. Generate recommendations

# Input is triples : person, paper, score
# This is (very) sparse data, so store it in a dictionary
# Turn this dictionary into a dense array

# Example

class Recommendations:

    def __init__(self, EPS):
        self.EPS = EPS # np.finfo.eps

    def prep_data(all_scores):

        # Names of all people in alphabetical order
        people = all_scores.key()
        people.sort()

        # Names of all papers in alphabetical order
        papers = set()
        for person in people:
            for title in all_scores[person].keys():
                papers.add(title)
        papers = list(papers)
        papers.sort()

        # Create and fill array
        ratings = np.zeros((len(people), len(papers)))
        for (person_id, person) in enumerate(people):
            for (title_id, title) in enumerate(papers):
                r = all_scores[person].get(title, 0)
                ratings[person_id, title_id] = float(r)

        return people, papers, ratings

# Next step is to compare sets of ratings
# Many ways to do this
# We will consider:
# - Inverse sums of squares
# - Pearson correlation coefficient
# Remember : 0 in matrix means "no rating"
# Doesn't make sense to compare ratings unless both people have read the paper
# Limit our metrics by masking the array

    def sim_distance(self, prefs, left_index, right_index):

        # Where do both people have preferences?
        left_has_prefs = prefs[left_index, :] > 0
        right_has_prefs = prefs[right_index, :] > 0
        mask = np.logical_and(left_has_prefs, right_has_prefs)

        # Not enough signal
        if np.sum(mask) < self.EPS:
            return 0

        # Return sum-of-squares distance
        diff = prefs[left_index, mask] - prefs[right_index, mask]
        sum_of_square = np.linalg.norm(diff) ** 2

        return 1/(1 + sum_of_square)

# What if two people rate many of the same papers but one always rates them lower than the other?
# If they rank papers the same, but use a different scale, we want to report
# that they rate papers the same way

# Pearson's Correlation reports the correlation between two individuals rather
# than the absolute difference.
# Pearson's Correlation Score measures the error of a best fit line between two individuals.
# To calculate Pearson's Correlation, we need to introduce two quantities:
# The standard deviation is the divergence from the mean:
# StDev(X) = E(X^2)-E(X)^2
# The covariance measures how two variables change together
# Cov(X,Y) = E(XY)-E(X)E(Y)
# Pearson's Correlation is:
# r=Cov(X,Y)/(StDev(X) * StDev(Y))
# Use NumPy to calculate both terms

# If a and b are N*1 arrays, then np.cov(a,b) returns an array of results
#     Variance(a)       Covariance(a,b)
#   Covariance(a,b)       Variance(a)
# Use it to calculate numerator and denominator

    def sim_pearson(self, prefs, left_index, right_index):

        # Where do both have ratings?
        rating_left = prefs[left_index, :]
        rating_right = prefs[right_index, :]
        mask = np.logical_and(rating_left > 0, rating_right > 0)

        # Summing over Booleans gives number of Trues
        num_common = np.sum(mask)

        # Return zero if there are no common ratings
        if num_common == 0:
            return 0

        # Caculate Pearson score "r"
        varcovar = np.cov(rating_left[mask], rating_right[mask])
        numerator = varcovar[0, 1]
        denominator = np.sqrt(varcovar[0, 0] * np.sqrt(varcovar[1, 1]))
        if denominator < self.EPS:
            return 0
        r = numerator / denominator
        return r

# Now that we have the scores we can:
# 1. Find people who rate papers most similarly
# 2. Find papers that are rated most similarly
# 3. Recommend papers for individuals based on the rankings of other people
#    and their similarity with this person's previous rankings
# To find individuals with the most similar ratings,
# apply a similarity algorithm to compare each person to every other person
# Sort the results to list most similar people first

    def top_matches(self, ratings, person, num, similarity):

        scores = []
        for other in range(ratings.shape[0]):
            if other != person:
                s = similarity(ratings, person, other)
                scores.append((s, other))
        scores.sort()
        scores.reverse()   # highest score should be the first

        return scores[0:num]

# Use the same idea to compute papers that are most similar
# Since both similarity functions compare rows of the data matrix,
# we must transpose it
# And change names to refer to papers, not people

    def similar_items(self, paper_ids, ratings, num = 10):
        result = {}
        ratings_by_paper = ratings.T
        for item in range(ratings_by_paper.shape[0]):
            temp = self.top_matches(ratings_by_paper, item, num, self.sim_distance)
            scores = []
            for (scores, name) in temp:
                scores.append((scores, paper_ids[name]))
            result[paper_ids[item]] = scores
        return result

# Finally suggest papers based on their rating by people
# who rated other papers similarly
# Recommendation score is the weighted average of paper scores,
# with weights assigned based on the similarity between individuals
# Only recommend papers that have not been rated yet

    def recommendations(self, prefs, person_id, similarity):

        totals, sim_sums = {}, {}
        num_people, num_papers = prefs.shape

        for other_id in range(num_people):
            # Don't compare people to themselves.
            if other_id == person_id:
                continue
            sim = similarity(prefs, person_id, other_id)
            if sim < self.EPS:
                continue

        for other_id in range(num_people):
            for title in range(num_papers):
                # Only score papers person hasn't seen yet
                if prefs[person_id, title] < self.EPS and \
                                prefs[other_id, title] > 0:
                    if title in totals:
                        totals[title] += sim * \
                                         prefs[other_id, title]
                    else:
                        totals[title] = 0
        # Create the normalized list
        rankings = []
        for title, total in totals.items():
            rankings.append((total/sim_sums[title], title))

        # Return the sorted list
        rankings.sort()
        rankings.reverse()  # highly recommended paper should be the first
        return rankings

# Major points:
# 1. Mathematical operations on matrix were all handled by NumPy
# 2. We still had to take care of data (re)formatting
'''