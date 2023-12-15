# Birds classification

This app classified birds by url photo.

Clone the repository, install requirements. Run the app with command 'uvicorn main:app'
And then use it like in example.

Example:
curl -X 'POST' 'http://127.0.0.1:8000/prediction/' -H 'Content-Type: application/json' -d '{"url": "https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_960,f_auto/DCTM_Penguin_UK_DK_AL526630_wkmzns.jpg"}'

* change url in your example
