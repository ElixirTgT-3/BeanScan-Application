package com.example.beanscan_application

import android.util.Log
import android.view.KeyEvent
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.EventChannel

class VolumeButtonsPlugin : FlutterPlugin, ActivityAware, EventChannel.StreamHandler {
    private var eventChannel: EventChannel? = null
    private var events: EventChannel.EventSink? = null
    private var binding: ActivityPluginBinding? = null
    private val channelName = "beanscan/volume_buttons"
    private val logTag = "BeanScanVolume"

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        Log.d(logTag, "Attaching to engine, setting up channel: $channelName")
        eventChannel = EventChannel(binding.binaryMessenger, channelName)
        eventChannel?.setStreamHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        Log.d(logTag, "Detaching from engine, tearing down channel")
        eventChannel?.setStreamHandler(null)
        eventChannel = null
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        Log.d(logTag, "Stream listen attached")
        this.events = events
    }

    override fun onCancel(arguments: Any?) {
        Log.d(logTag, "Stream listen cancelled")
        this.events = null
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        Log.d(logTag, "Attached to activity")
        this.binding = binding
        binding.addOnNewIntentListener { false }
    }

    override fun onDetachedFromActivityForConfigChanges() {
        Log.d(logTag, "Detached from activity for config change")
        binding = null
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        Log.d(logTag, "Reattached to activity after config change")
        this.binding = binding
    }

    override fun onDetachedFromActivity() {
        Log.d(logTag, "Detached from activity")
        binding = null
    }

    fun handleKeyEvent(keyCode: Int, event: KeyEvent?) {
        if (event?.action != KeyEvent.ACTION_DOWN) return
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d(logTag, "VOLUME_DOWN dispatched from plugin")
                events?.success("VOLUME_DOWN")
            }
            KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(logTag, "VOLUME_UP dispatched from plugin")
                events?.success("VOLUME_UP")
            }
        }
    }
}
