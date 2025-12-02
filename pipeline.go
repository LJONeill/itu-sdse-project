package main

import (
	"context"

	"dagger.io/dagger"
)

func main() {

	Build()
	//since this happens everytime, i think i should be able to reuse the python variable

	// maybe there should be a variable in the config file that changes after being run once instead?
	initial_config()

	repeated_config()

	data_pipeline()

	machine_learning_pipeline()

	plot_it_all()

}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("mlops_sg")).
		WithExec([]string{"python", "--version"})

	python = python.WithExec([]string{"python", "__init__.py"})

	return nil
}

func initial_config() {

	//runs the config that only happens once
	//directory structuring
	//i think this file is supposed to contain the os directory making code from the notebook

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("mlops_sg")).
		WithExec([]string{"python", "one_time_config.py"})

}

func repeated_config() {

	//config that needs to be present in every code run
	//variable defines
	//Path defines
	//package/requirements downloads??

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("mlops_sg")).
		WithExec([]string{"python", "repeated_config.py"})

}

func data_pipeline() {

	//if from a remote repository
	//get the raw data and put it in the appropriate folder
	//only run if data is known to have been updated/changed
	//contains data load/clean and feature manipulation

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("mlops_sg")).
		WithExec([]string{"python", "dataset.py"})

}

func machine_learning_pipeline() {

	//uses the features to train and output a model
	//only runs if data or model parameters have changed
	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("mlops_sg/modelling")).
		WithExec([]string{"python", "predict.py"})

	python = python.WithExec([]string{"python", "train.py"})

}

func plot_it_all() {

	//nothing about the model changes, this just gets plots
	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("mlops_sg")).
		WithExec([]string{"python", "plots.py"})

}
