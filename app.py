import streamlit as st
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
import xml.etree.ElementTree as ET
import folium
from streamlit_folium import st_folium
import plotly.express as px  # 新增：用于绘制可交互的图表

# ==========================================
# 1. 数据加载类 (DataLoader) - 保持不变
# ==========================================
class DataLoader:
    def __init__(self, file_content):
        self.file_content = file_content
        self.times = []
        self.distances = []
        self.lats = []
        self.lons = []

    # 用haversine 公式 计算两点之间的距离
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def parse(self):
        try:
            # 用 XML.etree.ElementTree 解析 GPX
            tree = ET.parse(self.file_content)
            root = tree.getroot()
            ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
            points = root.findall('.//gpx:trkpt', ns)
            if not points:
                 ns = {}
                 points = root.findall('.//trkpt')
            
            if not points: return None, None, None, None

            # 这里是为了兼容不同的格式
            parsed_data = []
            for trkpt in points:
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                time_elem = trkpt.find('gpx:time', ns) if ns else trkpt.find('time')
                
                if time_elem is not None:
                    t_str = time_elem.text.replace('Z', '')
                    try:
                        t_obj = datetime.fromisoformat(t_str)
                    except AttributeError:
                        if '.' in t_str:
                            t_obj = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%S.%f")
                        else:
                            t_obj = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                         continue
                    parsed_data.append((lat, lon, t_obj))

            if not parsed_data: return None, None, None, None

            start_time = parsed_data[0][2]
            total_dist = 0.0
            
            self.times = [0.0]
            self.distances = [0.0]
            self.lats = [parsed_data[0][0]]
            self.lons = [parsed_data[0][1]]

            # 计算每两个点之间的时间差和距离
            for i in range(1, len(parsed_data)):
                prev = parsed_data[i-1]
                curr = parsed_data[i]
                
                dt = (curr[2] - start_time).total_seconds()
                if dt <= self.times[-1]:
                    continue

                dist_step = self._haversine_distance(prev[0], prev[1], curr[0], curr[1])
                total_dist += dist_step
                
                self.times.append(dt)
                self.distances.append(total_dist)
                self.lats.append(curr[0])
                self.lons.append(curr[1])
                
            return np.array(self.times), np.array(self.distances), np.array(self.lats), np.array(self.lons)
        except Exception as e:
            st.error(f"解析错误: {e}")
            return None, None, None, None


class NumericalEngine:
    @staticmethod
    def calculate_velocity(time_arr, dist_arr):
        n = len(time_arr)
        v = np.zeros(n)
        for i in range(1, n - 1):
            h1 = time_arr[i] - time_arr[i-1]
            h2 = time_arr[i+1] - time_arr[i]
            if h1 > 0 and h2 > 0:
                # lagrange 插值后求导
                s_prev, s_curr, s_next = dist_arr[i-1], dist_arr[i], dist_arr[i+1]
                term1 = - (h2 / (h1 * (h1 + h2))) * s_prev
                term2 =   ((h2 - h1) / (h1 * h2)) * s_curr
                term3 =   (h1 / (h2 * (h1 + h2))) * s_next
                v[i] = term1 + term2 + term3
        if n >= 2:
            # 退化到一阶差商求导
            v[0] = (dist_arr[1]-dist_arr[0])/(time_arr[1]-time_arr[0])
            v[n-1] = (dist_arr[n-1]-dist_arr[n-2])/(time_arr[n-1]-time_arr[n-2])
        return v

    @staticmethod
    def calculate_integral_distance(time_arr, v_arr):
        n = len(time_arr)
        s_calc = np.zeros(n)
        current_s = 0.0
        for i in range(1, n):
            dt = time_arr[i] - time_arr[i-1]
            # 复化求积公式
            dS = (v_arr[i] + v_arr[i-1]) * dt / 2.0
            current_s += dS
            s_calc[i] = current_s
        return s_calc

    @staticmethod
    def calculate_metrics(v_arr, total_dist, total_time):
        avg_speed_kph = (total_dist / total_time * 3.6) if total_time > 0 else 0
        max_speed_kph = np.max(v_arr) * 3.6
        moving_mask = v_arr > 0.5
        moving_speed_kph = (np.mean(v_arr[moving_mask]) * 3.6) if np.any(moving_mask) else 0
        calories = (total_dist / 1000.0) * 25
        return avg_speed_kph, max_speed_kph, moving_speed_kph, calories

# 说明: 使用了 AI 辅助写前端，问过了是批准的

def main():
    st.set_page_config(page_title="数值分析大作业 - GPX 分析", layout="wide")
    
    st.sidebar.header("📂 数据与设置")
    uploaded_file = st.sidebar.file_uploader("上传 GPX 文件", type=["gpx"])
    
    st.title("🏃‍♂️ 运动轨迹数值分析系统")
    st.markdown("Project 4: Numerical Analysis of Motion Trajectory")

    if uploaded_file is not None:
        # 1. 预处理
        loader = DataLoader(uploaded_file)
        t_arr, s_real, lats, lons = loader.parse()
        
        if t_arr is None or len(t_arr) < 2:
            st.error("数据解析失败或数据点过少。")
            return

        # 2. 核心计算
        start_cpu = time.time()
        v_calc = NumericalEngine.calculate_velocity(t_arr, s_real)
        s_integrated = NumericalEngine.calculate_integral_distance(t_arr, v_calc)
        end_cpu = time.time()
        compute_time = (end_cpu - start_cpu) * 1000

        # 3. 指标计算
        avg_kph, max_kph, mov_kph, cal = NumericalEngine.calculate_metrics(v_calc, s_real[-1], t_arr[-1])
        final_real = s_real[-1]
        final_calc = s_integrated[-1]
        abs_error = abs(final_calc - final_real)
        rel_error = (abs_error / final_real) * 100 if final_real != 0 else 0

        # --- 核心指标看板 (2x3) ---
        st.subheader("📊 核心数据看板 (Key Metrics)")
        c1, c2, c3 = st.columns(3)
        c1.metric(label="🏁 原始路程 (GPS)", value=f"{final_real/1000:.3f} km")
        c2.metric(label="⏱️ 总时间", value=f"{t_arr[-1]/60:.1f} min", delta=f"{len(t_arr)} 采样点", delta_color="off")
        c3.metric(label="∫ 积分估算路程", value=f"{final_calc/1000:.3f} km", delta=f"误差 {rel_error:.4f}%", delta_color="inverse")
        
        st.divider()
        c4, c5, c6 = st.columns(3)
        c4.metric(label="🐢 平均速率", value=f"{avg_kph:.2f} km/h")
        c5.metric(label="🚴 移动速率 (Moving)", value=f"{mov_kph:.2f} km/h")
        c6.metric(label="🐇 最大速率", value=f"{max_kph:.2f} km/h")

        # --- 算法验证 (Plotly 交互图表) ---
        st.divider()
        st.subheader("🧪 算法验证: 原始路程 vs 积分路程 (交互版)")
        
        # 准备 Plotly 数据
        df_verify = pd.DataFrame({
            "Time (s)": t_arr,
            "Original GPS (m)": s_real,
            "Integrated Calc (m)": s_integrated
        })
        # 使用 Plotly 绘制双线
        fig_verify = px.line(df_verify, x="Time (s)", y=["Original GPS (m)", "Integrated Calc (m)"], 
                             color_discrete_map={"Original GPS (m)": "#2980B9", "Integrated Calc (m)": "#E67E22"})
        fig_verify.update_traces(mode="lines", hovertemplate="时间: %{x:.1f}s<br>路程: %{y:.2f}m") # 自定义悬停提示
        fig_verify.update_layout(hovermode="x unified", legend_title="数据来源") # 统一显示X轴信息
        st.plotly_chart(fig_verify, use_container_width=True)

        # --- 地图交互 (带悬停数据) ---
        st.divider()
        st.subheader("🗺️ 轨迹地图 (Hover for Info)")
        st.caption("注：鼠标悬停在轨迹点上，即可查看该点的瞬时速度、时间和路程。")
        
        # 初始化地图中心
        m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=14, tiles="CartoDB positron")
        
        # 1. 画轨迹底线 (蓝色，粗线) - 用于视觉概览
        coords = list(zip(lats, lons))
        folium.PolyLine(coords, color="#3498DB", weight=4, opacity=0.6).add_to(m)
        
        # 2. 添加交互点 (关键步骤)
        # 为了防止浏览器卡顿，如果点太多，我们需要降采样 (Downsampling)
        # 例如最多只显示 300 个交互点，均匀分布
        total_points = len(lats)
        max_interactive_points = 300 
        step = max(1, total_points // max_interactive_points)
        
        for i in range(0, total_points, step):
            # 构建悬停提示内容 (HTML 格式)
            tooltip_txt = f"""
            <div style="font-family: sans-serif; font-size: 12px;">
                <b>Time:</b> {t_arr[i]:.1f} s<br>
                <b>Dist:</b> {s_real[i]:.1f} m<br>
                <b>Speed:</b> {v_calc[i]*3.6:.1f} km/h
            </div>
            """
            
            folium.CircleMarker(
                location=[lats[i], lons[i]],
                radius=4,             # 半径适中，方便鼠标指到
                color='red',          # 边框颜色
                fill=True,
                fill_color='red',     # 填充颜色
                fill_opacity=0.0,     # 透明度设为0 (或者设为0.1)，这样看起来像是只有鼠标放上去才会有反应
                opacity=0.0,          # 边框也透明，实现“隐形触发区”
                tooltip=tooltip_txt   # 核心：悬停显示信息
            ).add_to(m)

        # 3. 起终点标记
        folium.Marker(coords[0], icon=folium.Icon(color='green', icon='play'), tooltip="Start").add_to(m)
        folium.Marker(coords[-1], icon=folium.Icon(color='red', icon='flag'), tooltip="End").add_to(m)

        st_folium(m, height=500, width=1000)
        
        # --- 速度曲线 (Plotly 交互版) ---
        st.divider()
        st.subheader("📈 速度曲线 (实时交互)")
        
        # 准备 Plotly 数据
        df_speed = pd.DataFrame({
            "Time (s)": t_arr,
            "Velocity (km/h)": v_calc * 3.6, # 转换为 km/h 显示更符合直觉
            "Velocity (m/s)": v_calc
        })
        
        fig_speed = px.line(df_speed, x="Time (s)", y="Velocity (km/h)", title="Instantaneous Velocity")
        fig_speed.update_traces(line_color='#C0392B', hovertemplate="时间: %{x:.1f}s<br>速度: %{y:.1f} km/h")
        fig_speed.update_layout(hovermode="x unified") # 鼠标移动时显示标尺
        
        # 添加一条平均速度参考线
        fig_speed.add_hline(y=mov_kph, line_dash="dot", annotation_text=f"Avg Moving: {mov_kph:.1f} km/h", annotation_position="top right")
        
        st.plotly_chart(fig_speed, use_container_width=True)

    else:
        st.info("👈 请上传 GPX 文件开始分析")

if __name__ == "__main__":
    main()