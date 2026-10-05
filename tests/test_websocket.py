"""WebSocket lifecycle regressions without GPU/model downloads."""
import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

class Processor:
    instances = []

    def __init__(self):
        self.closed = False
        self.cleanup_count = 0
        self.language = None
        self.parameters = None
        self.instances.append(self)

    def set_language(self, language):
        self.language = language

    def update_parameters(self, parameters):
        self.parameters = parameters

    async def cleanup(self):
        self.closed = True
        self.cleanup_count += 1

    async def process_audio_chunk(self, audio):
        if self.closed:
            raise RuntimeError('processor already closed')
        return [{'text': 'synthetic transcription'}]

stubs = {name: types.ModuleType(name) for name in ('stt', 'stt_openai', 'stt_whisperx')}
stubs['stt'].TranscriptionProcessor = Processor
with patch.dict(sys.modules, stubs):
    server = importlib.import_module('main')

class Socket:
    client = types.SimpleNamespace(host='127.0.0.1', port=1234)

    def __init__(self, backend, parameters=None):
        import json
        self.messages = iter([
            {'text': json.dumps({'type': 'config', 'backend': backend,
                                'language': 'ja', 'parameters': parameters})},
            {'bytes': b'\x00\x00'},
            {'type': 'websocket.disconnect'},
        ])
        self.sent = []
        self.receives = 0

    async def accept(self):
        pass

    async def receive(self):
        self.receives += 1
        return next(self.messages)

    async def send_json(self, data):
        self.sent.append(data)

class WebsocketLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        Processor.instances = []

    async def test_unavailable_openai_preserves_current_processor(self):
        await self.check_unavailable('openai-whisper')

    async def test_unavailable_whisperx_preserves_current_processor(self):
        await self.check_unavailable('whisperx')

    async def check_unavailable(self, backend):
        socket = Socket(backend)
        await server.websocket_endpoint(socket)
        self.assertEqual(socket.sent[-1], {'text': 'synthetic transcription'})
        self.assertIn('error', socket.sent[0])
        self.assertEqual(Processor.instances[0].cleanup_count, 1)
        self.assertEqual(server.active_connections, set())
        self.assertEqual(server.session_map, {})

    async def test_successful_switch_applies_configuration(self):
        socket = Socket('openai-whisper', {'beam_size': 3})
        with patch.object(server, 'OpenAIWhisperProcessor', Processor):
            await server.websocket_endpoint(socket)
        self.assertEqual(len(Processor.instances), 2)
        self.assertEqual([p.cleanup_count for p in Processor.instances], [1, 1])
        self.assertEqual(Processor.instances[-1].language, 'ja')
        self.assertEqual(Processor.instances[-1].parameters, {'beam_size': 3})
        self.assertEqual(socket.sent, [{'text': 'synthetic transcription'}])

    async def test_disconnect_event_ends_receive_loop(self):
        socket = Socket('faster-whisper')
        await server.websocket_endpoint(socket)
        self.assertEqual(socket.receives, 3)

    async def test_replacement_is_cleaned_up_when_old_cleanup_fails(self):
        class CleanupFailure(Processor):
            async def cleanup(self):
                await super().cleanup()
                raise RuntimeError('synthetic cleanup failure')

        socket = Socket('openai-whisper')
        with patch.object(server, 'TranscriptionProcessor', CleanupFailure), patch.object(
                server, 'OpenAIWhisperProcessor', Processor):
            await server.websocket_endpoint(socket)
        self.assertEqual(len(Processor.instances), 2)
        self.assertEqual(Processor.instances[-1].cleanup_count, 1)
        self.assertEqual(server.active_connections, set())

    async def test_failed_switch_preserves_current_processor(self):
        socket = Socket('whisperx')
        with patch.object(server, 'WhisperXProcessor', side_effect=RuntimeError('synthetic failure')):
            await server.websocket_endpoint(socket)
        self.assertEqual(socket.sent[-1], {'text': 'synthetic transcription'})
        self.assertEqual(len(Processor.instances), 1)
        self.assertEqual(Processor.instances[0].cleanup_count, 1)

if __name__ == '__main__':
    unittest.main()
