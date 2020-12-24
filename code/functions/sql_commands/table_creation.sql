
DROP TABLE IF EXISTS video_frame_vjump_pose CASCADE;
CREATE TABLE video_frame_vjump_pose (
  videofileName varchar(50) NOT NULL,
  frame_no smallint NOT NULL,
  model_name varchar(10) NOT NULL,
  model_version varchar(10) NOT NULL,
  detected_pose varchar(50),
  detected_pose_conf numeric(3,2) default 0,
  actual_pose varchar(50),
  PRIMARY KEY (videofileName, frame_no, model_name, model_version));
