commit bc19e9364b6c862ee28974245f09befdb4734748
Author: MelakuNe <mnegussie80@gmail.com>
Date:   Fri Jan 6 12:26:41 2023 +0300

    Ticket

diff --git a/run.py b/run.py
new file mode 100644
index 0000000..e96deac
--- /dev/null
+++ b/run.py
@@ -0,0 +1,5 @@
+import cv2 as cv
+
+path = '/train/all.jpg'
+image = cv.imread(path)
+centers, n_trains, scores = predict_image(image)
