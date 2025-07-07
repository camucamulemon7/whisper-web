#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper Real-time Transcription Usage Statistics Analyzer

This script analyzes the session logs to provide usage statistics.
"""

import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple
import json

def parse_log_line(line: str) -> Dict:
    """Parse a single log line for session information"""
    session_data = {}
    
    # SESSION_START pattern
    start_match = re.search(r'SESSION_START \| connection_id=([^\s]+) \| start_time=([^\s]+) \| total_connections=(\d+)', line)
    if start_match:
        session_data['type'] = 'start'
        session_data['connection_id'] = start_match.group(1)
        session_data['start_time'] = datetime.fromisoformat(start_match.group(2))
        session_data['total_connections'] = int(start_match.group(3))
        return session_data
    
    # SESSION_END pattern
    end_match = re.search(
        r'SESSION_END \| connection_id=([^\s]+) \| end_time=([^\s]+) \| duration_seconds=([\d.]+) \| '
        r'audio_source=([^\s]+) \| language=([^\s]+) \| backend=([^\s]+)', 
        line
    )
    if end_match:
        session_data['type'] = 'end'
        session_data['connection_id'] = end_match.group(1)
        session_data['end_time'] = datetime.fromisoformat(end_match.group(2))
        session_data['duration_seconds'] = float(end_match.group(3))
        session_data['audio_source'] = end_match.group(4)
        session_data['language'] = end_match.group(5)
        session_data['backend'] = end_match.group(6)
        return session_data
    
    return {}

def analyze_logs(log_file_path: str):
    """Analyze log file and generate statistics"""
    sessions = {}
    
    # Read and parse log file
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'SESSION_' in line:
                    data = parse_log_line(line)
                    if data:
                        if data['type'] == 'start':
                            sessions[data['connection_id']] = {
                                'start_time': data['start_time'],
                                'total_connections': data['total_connections']
                            }
                        elif data['type'] == 'end':
                            if data['connection_id'] in sessions:
                                sessions[data['connection_id']].update({
                                    'end_time': data['end_time'],
                                    'duration_seconds': data['duration_seconds'],
                                    'audio_source': data['audio_source'],
                                    'language': data['language'],
                                    'backend': data['backend']
                                })
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading log file: {e}")
        return
    
    # Generate statistics
    completed_sessions = [s for s in sessions.values() if 'duration_seconds' in s]
    
    if not completed_sessions:
        print("No completed sessions found in the log file.")
        return
    
    # Basic statistics
    total_sessions = len(completed_sessions)
    total_duration = sum(s['duration_seconds'] for s in completed_sessions)
    avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
    
    # Group by date
    sessions_by_date = defaultdict(list)
    for session in completed_sessions:
        date = session['start_time'].date()
        sessions_by_date[date].append(session)
    
    # Group by audio source
    sessions_by_source = defaultdict(list)
    for session in completed_sessions:
        sessions_by_source[session.get('audio_source', 'unknown')].append(session)
    
    # Group by language
    sessions_by_language = defaultdict(list)
    for session in completed_sessions:
        sessions_by_language[session.get('language', 'unknown')].append(session)
    
    # Group by backend
    sessions_by_backend = defaultdict(list)
    for session in completed_sessions:
        sessions_by_backend[session.get('backend', 'unknown')].append(session)
    
    # Print report
    print("=" * 80)
    print("WHISPER REAL-TIME TRANSCRIPTION - USAGE STATISTICS")
    print("=" * 80)
    print()
    
    print(f"📊 OVERALL STATISTICS")
    print(f"  Total Sessions: {total_sessions}")
    print(f"  Total Duration: {format_duration(total_duration)}")
    print(f"  Average Duration: {format_duration(avg_duration)}")
    print()
    
    print(f"📅 DAILY USAGE")
    for date in sorted(sessions_by_date.keys()):
        daily_sessions = sessions_by_date[date]
        daily_duration = sum(s['duration_seconds'] for s in daily_sessions)
        print(f"  {date}: {len(daily_sessions)} sessions, {format_duration(daily_duration)}")
    print()
    
    print(f"🎤 BY AUDIO SOURCE")
    for source, source_sessions in sessions_by_source.items():
        source_duration = sum(s['duration_seconds'] for s in source_sessions)
        percentage = (len(source_sessions) / total_sessions) * 100
        print(f"  {source}: {len(source_sessions)} sessions ({percentage:.1f}%), {format_duration(source_duration)}")
    print()
    
    print(f"🌐 BY LANGUAGE")
    for lang, lang_sessions in sessions_by_language.items():
        lang_duration = sum(s['duration_seconds'] for s in lang_sessions)
        percentage = (len(lang_sessions) / total_sessions) * 100
        print(f"  {lang}: {len(lang_sessions)} sessions ({percentage:.1f}%), {format_duration(lang_duration)}")
    print()
    
    print(f"⚙️  BY BACKEND")
    for backend, backend_sessions in sessions_by_backend.items():
        backend_duration = sum(s['duration_seconds'] for s in backend_sessions)
        percentage = (len(backend_sessions) / total_sessions) * 100
        print(f"  {backend}: {len(backend_sessions)} sessions ({percentage:.1f}%), {format_duration(backend_duration)}")
    print()
    
    # Find longest and shortest sessions
    if completed_sessions:
        longest = max(completed_sessions, key=lambda s: s['duration_seconds'])
        shortest = min(completed_sessions, key=lambda s: s['duration_seconds'])
        
        print(f"📈 NOTABLE SESSIONS")
        print(f"  Longest: {format_duration(longest['duration_seconds'])} on {longest['start_time'].date()}")
        print(f"  Shortest: {format_duration(shortest['duration_seconds'])} on {shortest['start_time'].date()}")
    
    print()
    print("=" * 80)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_usage.py <log_file_path>")
        print("Example: python analyze_usage.py /path/to/backend.log")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    analyze_logs(log_file_path)

if __name__ == "__main__":
    main()
