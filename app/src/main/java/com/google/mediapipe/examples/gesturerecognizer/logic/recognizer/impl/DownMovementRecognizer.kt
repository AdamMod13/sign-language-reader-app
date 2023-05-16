package com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.impl

import android.util.Log
import com.google.mediapipe.examples.gesturerecognizer.logic.model.GestureWrapper
import com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.DynamicGestureRecognizer


class DownMovementRecognizer : DynamicGestureRecognizer {
    override fun checkHandMovement(gestureList: List<GestureWrapper>, landmarkIndex: Int): Boolean {
        if (gestureList.size < minLength)
            return false
        val movements = mutableListOf<Float>()
        for (i in gestureList.size - 1 downTo 1) {
            val movement = comparePositions(gestureList[i], gestureList[i - 1], landmarkIndex)
            Log.i("checkHandMovement", "$movement between indexed $i and ${i - 1}")
            movements.add(movement)
        }
        return movements.all { it > movementThreshold }
    }

    private fun comparePositions(
        first: GestureWrapper, second: GestureWrapper, landmarkIndex: Int
    ): Float {
        return first.getLandmarkArray()[landmarkIndex].y() - second.getLandmarkArray()[landmarkIndex].y()
    }
}