package com.kyhsgeekcode.minecraftenv.mixin;

/*
 * This file is part of the Gamma Utils project and is licensed under the GNU Lesser General Public License v3.0.
 *
 * Copyright (C) 2021 Sjouwer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

import com.mojang.serialization.Codec;
import net.minecraft.client.option.SimpleOption;
import net.minecraft.client.resource.language.I18n;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(SimpleOption.class)
public class GammaMixin<T> {

  @Shadow @Final Text text;

  @Shadow T value;

  /** Mixin to allow saving "invalid" gamma values into the options file */
  @Inject(method = "getCodec", at = @At("HEAD"), cancellable = true)
  private void returnFakeCodec(CallbackInfoReturnable<Codec<Double>> info) {
    if (text.getString().equals(I18n.translate("options.gamma"))) {
      info.setReturnValue(Codec.DOUBLE);
    }
  }

  /** Mixin to allow setting "invalid" gamma values */
  @Inject(method = "setValue", at = @At("HEAD"), cancellable = true)
  private void setRealValue(T value, CallbackInfo info) {
    if (text.getString().equals(I18n.translate("options.gamma"))) {
      this.value = value;
      info.cancel();
    }
  }
}
