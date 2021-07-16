/*
 * cell.hpp
 *
 *  Created on: 12 Mar 2020
 *      Author: Prinzessin
 */

#ifndef INC_CELL_HPP_
#define INC_CELL_HPP_

enum Shape {UNKNOWN, ROUND_ALONE, TWO_ROUND, ROUND_SOCIAL, WEIRD_ALONE, WEIRD_SOCIAL};

/**
 * Single Cell Object
 */
class cell{
	public:
		Mat image;
		Mat mask;
		Rect roi; // x, y, width, height
		Shape shape = UNKNOWN;
};


#endif /* INC_CELL_HPP_ */
