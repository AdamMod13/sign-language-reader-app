package com.google.mediapipe.examples.gesturerecognizer.logic.recognizer

import com.google.mediapipe.examples.gesturerecognizer.logic.model.GestureWrapper

// todo: idea -> don't make it a interface -> pass a function to a method to determine movement (no copy-paste)
//                                                                                              (or maybe in other place ?)

interface DynamicGestureRecognizer {
    val minLength: Int get() = 4
    val movementThreshold: Float get() = 0.001f

    fun checkHandMovement(gestureList: List<GestureWrapper>): Boolean
}