def calc_IoU(rectA, rectB):
	xend_a = rectA[0] + rectA[2]
	yend_a = rectA[1] + rectA[3]
	xend_b = rectB[0] + rectA[2]
	yend_b = rectA[1] + rectA[3]
	outer_x = min(rectA[0], rectB[0])
	outer_y = min(rectA[1], rectB[1])
	outer_xend = max(xend_a, xend_b)
	outer_yend = max(yend_a, yend_b)
	outer_area = (outer_xend - outer_x) * (outer_yend - outer_y)
	inner_x = max(rectA[0], rectB[0])
	inner_y = max(rectA[1], rectB[1])
	inner_xend = min(xend_a, xend_b)
	inner_yend = min(yend_a, yend_b)
	inner_area = (inner_xend - inner_x) * (inner_yend - inner_y)
	return inner_area / float(outer_area)

print(calc_IoU([30, 30, 30, 30], [40, 40, 20, 20]))
