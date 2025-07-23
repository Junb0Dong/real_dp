from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

# 计算两个旋转姿态的距离（弧度）
def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

"""
1. 平滑插值：通过线性插值（位置）和 Slerp（姿态）生成平滑轨迹，避免机器人运动卡顿；
2. 可行性保证：通过 drive_to_waypoint 和 schedule_waypoint 确保轨迹不超过最大速度（不会 “要求机器人做不到的运动”）；
3. 动态调整：支持修剪轨迹、插入新航点，适应实时调整需求；
4. 通用性：同时支持静态位姿（单航点）和动态轨迹（多航点）。
"""

class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:,:3]
            rot = st.Rotation.from_rotvec(poses[:,3:])

            self.pos_interp = si.interp1d(times, pos, 
                axis=0, assume_sorted=True) # 位置插值
            self.rot_interp = st.Slerp(times, rot)  # 旋转插值
    
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:,:3] = self.pos_interp.y
            poses[:,3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    # 截取轨迹的一部分
    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)
    
    def drive_to_waypoint(self, pose, time, curr_time, max_pos_speed=np.inf, max_rot_speed=np.inf) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time) # 获取当前时刻的位姿
        pos_dist, rot_dist = pose_distance(curr_pose, pose) # 计算两个位姿之间的距离
        pos_min_duration = pos_dist / max_pos_speed # 平移所需最少时间
        rot_min_duration = rot_dist / max_rot_speed # 旋转所需最少时间
        duration = time - curr_time # 计算当前时刻到目标时刻的时间差
        duration = max(duration, max(pos_min_duration, rot_min_duration))   # 必须同时满足平移和旋转的时间要求
        assert duration >= 0
        last_waypoint_time = curr_time + duration   # 最终到达时间

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)    # 截取当前时间轨迹，过滤掉所有无关历史数据，只保留当前状态
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)   # 从curr_time到last_waypoint_time
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses) # 返回新的插值器（新轨迹）
        return final_interp

    def schedule_waypoint(self,
            pose, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp


    # 给定任意时间 t，返回该时刻的位姿
    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:,:3] = self.pos_interp(t)
            pose[:,3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
