"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy
from gspan_mining import GraphDatabase
from gspan_mining import gSpan
from sklearn import metrics, tree


class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """

    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []

        self.database = database  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print("Please implement the store function in a subclass for a specific mining task!")

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print("Please implement the prune function in a subclass for a specific mining task!")


class FrequentPositiveGraphs(PatternGraphs):
    """
    Finds the frequent (support >= minsup) subgraphs among the positive graphs.
    This class provides a method to build a feature matrix for each subset.
    """

    def __init__(self, minsup, database, subsets):
        """
        Initialize the task.
        :param minsup: the minimum positive support
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)
        self.patterns = []  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        self.patterns.append((dfs_code, gid_subsets))

    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        return len(gid_subsets[0]) < self.minsup

    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    # return a feature matrix for each subset of examples, in which the columns correspond to patterns
    # and the rows to examples in the subset.
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for pattern, gid_subsets in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]


class TopKConfident2(FrequentPositiveGraphs):
    def __init__(self, minsup, database, subsets, k):
        super().__init__(minsup, database, subsets)
        self.top = []
        self.k = k

    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        return len(gid_subsets[0] + gid_subsets[1]) < self.minsup

    def store(self, dfs_code, gid_subsets):
        total_support = len(gid_subsets[0]) + len(gid_subsets[1])
        confidence = len(gid_subsets[0]) / total_support
        if len(self.top) < self.k or confidence >= self.top[self.k - 1][0]:
            found = False
            for i, t in enumerate(self.top):
                if t[0] == confidence and t[1] == total_support:
                    t[2].append(dfs_code)
                    found = True
            if not found:
                self.top.append((confidence, total_support, [dfs_code]))
                self.top = sorted(self.top, reverse=True, key=lambda x: (x[0], x[1]))
                if len(self.top) > self.k:
                    del self.top[-1]
        self.patterns.append((dfs_code, gid_subsets))


class TopKConfident4(FrequentPositiveGraphs):
    def __init__(self, minsup, database, subsets, k):
        super().__init__(minsup, database, subsets)
        self.top = []
        self.k = k

    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        return len(gid_subsets[0] + gid_subsets[2]) < self.minsup

    def store(self, dfs_code, gid_subsets):
        total_support = len(gid_subsets[0]) + len(gid_subsets[2])
        confidence = len(gid_subsets[0]) / total_support
        if len(self.top) < self.k or confidence >= self.top[self.k - 1][0]:
            found = False
            for i, t in enumerate(self.top):
                if t[0] == confidence and t[1] == total_support:
                    t[2].append(dfs_code)
                    found = True
            if not found:
                self.top.append((confidence, total_support, [dfs_code]))
                self.top = sorted(self.top, reverse=True, key=lambda x: (x[0], x[1]))
                if len(self.top) > self.k:
                    del self.top[-1]
        self.patterns.append((dfs_code, gid_subsets))


class TopKConfident4Rule(TopKConfident4):
    def __init__(self, minsup, database, subsets):
        super().__init__(minsup, database, subsets, 1)
        self.patterns_dict = {}
        self.test_pos = {}
        self.test_neg = {}

    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        return len(gid_subsets[0] + gid_subsets[2]) < self.minsup

    def store(self, dfs_code, gid_subsets):
        for gid in gid_subsets[1]:
            try:
                self.test_pos[gid].append(dfs_code)
            except KeyError:
                self.test_pos[gid] = []
                self.test_pos[gid].append(dfs_code)

        for gid in gid_subsets[3]:
            try:
                self.test_neg[gid].append(dfs_code)
            except KeyError:
                self.test_neg[gid] = []
                self.test_neg[gid].append(dfs_code)

        total_support = len(gid_subsets[0]) + len(gid_subsets[2])
        confidence = max(len(gid_subsets[0]), len(gid_subsets[2])) / total_support
        if len(self.top) < self.k or confidence >= self.top[self.k - 1][0]:
            found = False
            for i, t in enumerate(self.top):
                if t[0] == confidence and t[1] == total_support:
                    t[2].append(dfs_code)
                    found = True
            if not found:
                self.top.append((confidence, total_support, [dfs_code]))
                self.top = sorted(self.top, reverse=True, key=lambda x: (x[0], x[1]))
                if len(self.top) > self.k:
                    del self.top[-1]
        self.patterns.append((dfs_code, gid_subsets))
        self.patterns_dict[dfs_code] = gid_subsets


def task1():
    """
    Runs gSpan with the specified positive and negative graphs, finds all frequent subgraphs in the positive class
    with a minimum positive support of minsup and prints them.
    """

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])  # Fourth parameter: minimum support

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
    task = TopKConfident2(minsup, graph_database, subsets, k)  # Creating task

    gSpan(task).run()  # Running gSpan

    for t in task.top:
        for pattern in t[2]:
            print('{} {} {}'.format(pattern, t[0], t[1]))


def task2():
    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])  # Fourth parameter: minimum support
    nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate(minsup, graph_database, subsets, k)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),
                # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),
                # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i + 1))
            train_and_evaluate(minsup, graph_database, subsets, k)


def task3():
    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])  # Fourth parameter: minimum support
    nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        sequential_rule_learning(minsup, graph_database, subsets, k)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),
                # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),
                # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i + 1))
            sequential_rule_learning(minsup, graph_database, subsets, k)


def train_and_evaluate(minsup, database, subsets, k):
    task = TopKConfident4(minsup, database, subsets, k)  # Creating task

    gSpan(task).run()  # Running gSpan

    # Creating feature matrices for training and testing:
    features = task.get_feature_matrices()
    train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
    train_labels = numpy.concatenate(
        (numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
    test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
    test_labels = numpy.concatenate(
        (numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

    classifier = tree.DecisionTreeClassifier(random_state=1)  # Creating model object
    classifier.fit(train_fm, train_labels)  # Training model

    predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

    accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:

    # Printing frequent patterns along with their positive support:
    for t in task.top:
        for pattern in t[2]:
            print('{} {} {}'.format(pattern, t[0], t[1]))
    # printing classification results:
    print(predicted.tolist())
    print('accuracy: {}'.format(accuracy))
    print()  # Blank line to indicate end of fold.


def sequential_rule_learning(minsup, database, subsets, k):
    ignored = []
    rules = []
    test_pos = []
    test_neg = []

    new_subsets = []
    for subset in subsets:
        new_subsets.append(list(subset))

    for i in range(k):
        to_del = []
        for j, subset in enumerate(new_subsets):
            if j == 0 or j == 2:
                for k, gid in enumerate(subset):
                    if int(gid) in ignored or gid in ignored:
                        to_del.append((j, k))

        to_del = sorted(to_del, reverse=True, key=lambda x: (x[0], x[1]))

        for t in to_del:
            del new_subsets[t[0]][t[1]]

        if len(new_subsets[0]) == 0 and len(new_subsets[2]) == 0:
            break

        task = TopKConfident4Rule(minsup, database, new_subsets)
        gSpan(task).run()

        if len(test_pos) == 0:
            test_pos = sorted(task.test_pos.items())
            test_neg = sorted(task.test_neg.items())

        try:
            top_pattern = min(task.top[0][2])
        except IndexError:
            break
        print('{} {} {}'.format(top_pattern, task.top[0][0], task.top[0][1]))
        cover = task.patterns_dict[top_pattern]
        rules.append((top_pattern, -1 if len(cover[2]) > len(cover[0]) else 1))
        newly_ignored = [task.patterns_dict[min(task.top[0][2])][0], task.patterns_dict[min(task.top[0][2])][2]]
        ignored.extend([item for sublist in newly_ignored for item in sublist])

    nb_pos = len(new_subsets[0])
    nb_neg = len(new_subsets[2])
    default_class = -1 if nb_neg > nb_pos else 1

    predictions = []
    nb_matches = 0
    for trans in test_pos:
        found = False
        for rule in rules:
            if rule[0] in trans[1]:
                predictions.append(rule[1])
                found = True
                break
        if not found:
            predictions.append(default_class)
        if predictions[-1] == 1:
            nb_matches += 1

    for trans in test_neg:
        found = False
        for rule in rules:
            if rule[0] in trans[1]:
                predictions.append(rule[1])
                found = True
                break
        if not found:
            predictions.append(default_class)
        if predictions[-1] == -1:
            nb_matches += 1

    print(predictions)
    print("accuracy: {}\n".format(nb_matches / len(predictions)))


if __name__ == '__main__':
    task3()
    # sys.stdout = open('test.txt', 'w')
    # task2()
    # sys.stdout = sys.__stdout__
