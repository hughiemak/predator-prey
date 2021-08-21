# Created by Hei Yi Mak 2021 

import collections

Experience = collections.namedtuple("Experience", field_names=['state', 'action', 'reward', 'done', 'next_state'])
Feedback_Experience = collections.namedtuple("Feedback_Experience", field_names=['state', 'action', 'reward', 'done', 'next_state'])
HER_Experience = collections.namedtuple("HER_Experience", field_names=['state', 'action', 'reward', 'done', 'next_state', 'goal'])