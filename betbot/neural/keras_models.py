"""LICENSE
Copyright 2020 Hermann Krumrey <hermann@krumreyh.com>

This file is part of betbot.

betbot is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

betbot is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with betbot.  If not, see <http://www.gnu.org/licenses/>.
LICENSE"""

from keras.models import Sequential
from keras.layers import Dense, Flatten


def table_history_model() -> Sequential:
    """
    Specifies a keras model to predict match results based on historical
    league table data
    :return: The keras model
    """
    model = Sequential()
    model.add(Flatten(input_shape=(12,)))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
