#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include "BlobLabeling.h"

int main()
{
	//IplImage* image = 0; //openCV에서 사용되는 자료형이다.

	IplImage* m_pImage = 0; // 변환될 이미지


	CvCapture* capture = cvCaptureFromCAM(0); //현재 인식된 웹캠을 찾고,
	cvNamedWindow("OpenCvCamtest", 0); // 화면을 그려줄 윈도우를 생성한다.

	cvResizeWindow("OpenCvCamtest", 640, 480); // 사이즈를 조절한다.(lpIImage를 할당하면서도 조절가능)

	while (1) {
		cvGrabFrame(capture);
		m_pImage = cvRetrieveFrame(capture); // 현재 인식된 장면을 받아오고image에 넣는다.

		//이진화
		IplImage* gray = cvCreateImage(cvSize(640, 480), 8, 1);

		cvCvtColor(m_pImage, gray, CV_BGR2GRAY);

		cvThreshold(gray, gray, 0.0, 255.0, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
		// 레이블링시 흰색(255)을 개체로 잡으므로 이진화된 영상을 반전시킨다.
		cvThreshold(gray, gray, 1, 255, CV_THRESH_BINARY_INV);

		// 레이블 결과 표시 위한 이미지 설정
		IplImage* labeled = cvCreateImage(cvSize(gray->width, gray->height), 8, 3);

		cvCvtColor(gray, labeled, CV_GRAY2BGR);

		// 레이블링
		CBlobLabeling blob;
		blob.SetParam(gray, 100);   // 레이블링 할 이미지와 최소 픽셀수 등을 설정

		blob.DoLabeling();


		int nMaxWidth = gray->width * 8 / 10;   // 영상 가로 전체 크기의 80% 이상인 레이블은 제거
		int nMaxHeight = gray->height * 8 / 10;   // 영상 세로 전체 크기의 80% 이상인 레이블은 제거

		blob.BlobSmallSizeConstraint(30, 30);
		blob.BlobBigSizeConstraint(nMaxWidth, nMaxHeight);

		for (int i = 0; i < blob.m_nBlobs; i++)
		{
			bool bIsMarker = false;
			CvPoint   pt1 = cvPoint(blob.m_recBlobs[i].x,
				blob.m_recBlobs[i].y);
			CvPoint pt2 = cvPoint(pt1.x + blob.m_recBlobs[i].width,
				pt1.y + blob.m_recBlobs[i].height);

			/////////////////////////////////////////////////////////////////////
			// 각 레이블이 속해있는 영상을 반전시킨 뒤, 
			// 그 영상 내부에서 다시 레이블을 찾아낸다.

			// 이미지 관심영역 설정
			cvSetImageROI(gray, blob.m_recBlobs[i]);

			IplImage* sub_gray = cvCreateImage(cvSize(blob.m_recBlobs[i].width, blob.m_recBlobs[i].height), 8, 1);

			blob.GetBlobImage(sub_gray, i);
			cvThreshold(gray, sub_gray, 1, 255, CV_THRESH_BINARY_INV);

			// 관심영역 해제
			cvResetImageROI(gray);

			////////////////////////////
			// 레이블링
			CBlobLabeling inner;
			inner.SetParam(sub_gray, 100);

			inner.DoLabeling();

			int nSubMinWidth = sub_gray->width * 4 / 10;
			int nSubMinHeight = sub_gray->height * 4 / 10;

			int nSubMaxWidth = sub_gray->width * 8 / 10;
			int nSubMaxHeight = sub_gray->height * 8 / 10;

			inner.BlobSmallSizeConstraint(nSubMinWidth, nSubMinHeight);
			inner.BlobBigSizeConstraint(nSubMaxWidth, nSubMaxHeight);

			for (int j = 0; j < inner.m_nBlobs; j++)
			{
				int nThick = 20;

				if (inner.m_recBlobs[j].x < nThick
					|| inner.m_recBlobs[j].y < nThick
					|| (sub_gray->width - (inner.m_recBlobs[j].x + inner.m_recBlobs[j].width)) < nThick
					|| (sub_gray->height - (inner.m_recBlobs[j].y + inner.m_recBlobs[j].height)) < nThick)   continue;

				CvPoint   s_pt1 = cvPoint(pt1.x + inner.m_recBlobs[j].x,
					pt1.y + inner.m_recBlobs[j].y);
				CvPoint s_pt2 = cvPoint(s_pt1.x + inner.m_recBlobs[j].width,
					s_pt1.y + inner.m_recBlobs[j].height);

				// green
				CvScalar green = cvScalar(0, 255, 0);

				cvDrawRect(labeled, s_pt1, s_pt2, green, 2);
				bIsMarker = true;

				break;
			}

			cvReleaseImage(&sub_gray);

			if (bIsMarker)
			{
				// 각 레이블 표시
				CvScalar red = cvScalar(0, 0, 255);

				cvDrawRect(labeled, pt1, pt2, red, 1);
				CvMemStorage* storage = cvCreateMemStorage(0);
				CvSeq* contours = 0;

				// 1. 윤곽 검출
				int mw = blob.m_recBlobs[i].width;
				int mh = blob.m_recBlobs[i].height;

				IplImage* marker = cvCreateImage(cvSize(mw, mh), 8, 1);

				blob.GetBlobImage(marker, i);

				cvFindContours(marker, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

				// 2. 윤곽선 표시 및 윤곽선 영상 생성
				//            cvDrawContours( m_pImage, contours, CV_RGB (255, 255, 0), CV_RGB (0, 255, 0), -1, 1, CV_AA, cvPoint (blob.m_recBlobs[i].x, blob.m_recBlobs[i].y));

				// 3. 꼭지점 추출
				int x;

				double fMaxDist;

				CvPoint      corner[4];

				for (x = 0; x < 4; x++)      corner[x] = cvPoint(0, 0);

				// 초기 위치 설정
				CvPoint *st = (CvPoint *)cvGetSeqElem(contours, 0);

				// 첫 번 째 꼭지점 추출(최대 거리를 가지는 점 선택)
				fMaxDist = 0.0;

				for (x = 1; x < contours->total; x++)
				{
					CvPoint* pt = (CvPoint *)cvGetSeqElem(contours, x);

					double fDist = sqrt((double)((st->x - pt->x) * (st->x - pt->x) + (st->y - pt->y) * (st->y - pt->y)));

					if (fDist > fMaxDist)
					{
						corner[0] = *pt;

						fMaxDist = fDist;
					}
				}

				// 두 번 째 꼭지점 추출(첫 번 째 꼭지점에서 최대 거리를 가지는 점 선택)
				fMaxDist = 0.0;

				for (x = 0; x < contours->total; x++)
				{
					CvPoint* pt = (CvPoint *)cvGetSeqElem(contours, x);

					double fDist = sqrt((double)((corner[0].x - pt->x) * (corner[0].x - pt->x) + (corner[0].y - pt->y) * (corner[0].y - pt->y)));

					if (fDist > fMaxDist)
					{
						corner[1] = *pt;

						fMaxDist = fDist;
					}
				}

				// 세 번 째 꼭지점 추출(첫 번 째, 두 번 째 꼭지점에서 최대 거리를 가지는 점 선택)
				fMaxDist = 0.0;

				for (x = 0; x < contours->total; x++)
				{
					CvPoint* pt = (CvPoint *)cvGetSeqElem(contours, x);

					double fDist = sqrt((double)((corner[0].x - pt->x) * (corner[0].x - pt->x) + (corner[0].y - pt->y) * (corner[0].y - pt->y)))
						+ sqrt((double)((corner[1].x - pt->x) * (corner[1].x - pt->x) + (corner[1].y - pt->y) * (corner[1].y - pt->y)));

					if (fDist > fMaxDist)
					{
						corner[2] = *pt;

						fMaxDist = fDist;
					}
				}

				// 네 번 째 꼭지점 추출
				// (벡터 내적을 이용하여 좌표평면에서 사각형의 너비의 최대 값을 구한다.)
				//                                           thanks to 송성원
				int x1 = corner[0].x;   int y1 = corner[0].y;
				int x2 = corner[1].x;   int y2 = corner[1].y;
				int x3 = corner[2].x;   int y3 = corner[2].y;

				int nMaxDim = 0;

				for (x = 0; x < contours->total; x++)
				{
					CvPoint* pt = (CvPoint *)cvGetSeqElem(contours, x);

					int x = pt->x;
					int y = pt->y;

					int nDim = abs((x1 * y2 + x2 * y + x  * y1) - (x2 * y1 + x  * y2 + x1 * y))
						+ abs((x1 * y + x  * y3 + x3 * y1) - (x  * y1 + x3 * y + x1 * y3))
						+ abs((x  * y2 + x2 * y3 + x3 * y) - (x2 * y + x3 * y2 + x  * y3));

					if (nDim > nMaxDim)
					{
						corner[3] = *pt;

						nMaxDim = nDim;
					}
				}
				CvFont font;
				cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, 8);

			

				// 모서리를 잇는 직선(BLUE)
				for (int m = 0; m < 3; m++)
					for (int n = m + 1; n < 4; n++)
					{
						int x1 = corner[m].x + blob.m_recBlobs[i].x;
						int y1 = corner[m].y + blob.m_recBlobs[i].y;
						int x2 = corner[n].x + blob.m_recBlobs[i].x;
						int y2 = corner[n].y + blob.m_recBlobs[i].y;

						cvLine(m_pImage, cvPoint(x1, y1), cvPoint(x2, y2), CV_RGB(0, 0, 255), 1);
					}

				for (int m = 0; m < 4; m++)
				{
					int x = corner[m].x + blob.m_recBlobs[i].x;
					int y = corner[m].y + blob.m_recBlobs[i].y;

					cvCircle(m_pImage, cvPoint(x, y), 2, CV_RGB(0, 0, 255), 2);

									
				}
			}


			cvShowImage("OpenCvCamtest", m_pImage); // image에 있는 장면을 윈도우에 그린다.

			if (cvWaitKey(5) >= 0) // 이게 가장 중요한데 이 WaitKey 함수가 없으면 아무 것도 안그린다.
				break;
		}
	}

	cvReleaseCapture(&capture); // 할당받았던 웹캠을 해제하고,
	cvDestroyWindow("OpenCvCamtest"); // 윈도우를 종료한다. 

	return 0;
}
