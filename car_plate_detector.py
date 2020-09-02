
import cv2

input = 'car3.jpg'
image = cv2.imread(input)



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
d, sigmaColor, sigmaSpace = 11, 17, 17
filtered_img = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)


lower, upper = 170, 200
edged = cv2.Canny(filtered_img, lower, upper)


cnts, hir = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts))
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
NumberPlateCnt = None

count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)

    epsilon = 0.01 * peri

    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) == 4:
        print(approx)
        NumberPlateCnt = approx
        break


cv2.imshow("Input Image", image)

cv2.imshow("Gray scale Image", gray)

cv2.imshow("After Applying Bilateral Filter", filtered_img)

cv2.imshow("After Canny Edges", edged)

cv2.drawContours(image, [NumberPlateCnt], -1, (255, 0, 0), 2)
cv2.imshow("Output", image)

cv2.waitKey(0)