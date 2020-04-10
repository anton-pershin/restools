from abc import ABC, abstractmethod


class DataAccessStrategy(ABC):
    """
    Class DataAccessStrategy is a base class for the data access strategy (see Strategy pattern for details). It is
    supposed to regulate whether the data being loaded by any means is going to be cached in a dedicated storage or not.
    """
    @abstractmethod
    def access_data(self, storage, data_id, upload_func, *upload_func_args):
        """
        Calls the function for data uploading (upload_func) and returns the resulting object. In the meanwhile, the
        object may or may not be stored in storage depending on the implementation in derived classes.

        :param storage: a object with implemented operator '[]' supposed to store data by index data_id
        :param data_id: data index (may be string, integer, whatever)
        :param upload_func: time-consuming function uploading data into temporary object returned by this function
        :param upload_func_args: arguments to be passed to upload_func
        :return: whatever upload_func returns or None
        """
        raise NotImplementedError('Must be implemented!')


class FreeDataAfterAccessStrategy(DataAccessStrategy):
    """
    Class FreeDataAfterAccessStrategy implements DataAccessStrategy so that no data is cached in storage
    """
    def access_data(self, storage, data_id, upload_func, *upload_func_args):
        data = None
        if data_id in storage:
            data = storage[data_id]
            del storage[data_id]
        else:
            data = upload_func(*upload_func_args)
        return data


class HoldDataInMemoryAfterAccessStrategy(DataAccessStrategy):
    """
    Class HoldDataInMemoryAfterAccessStrategy implements DataAccessStrategy so that all data is cached in storage
    """
    def access_data(self, storage, data_id, upload_func, *upload_func_args):
        data = None
        if data_id in storage:
            data = storage[data_id]
        else:
            data = upload_func(*upload_func_args)
            storage[data_id] = data
        return data


# These objects need to be created only once (sort of singletons) -- do it here
free_data_after_access_strategy = FreeDataAfterAccessStrategy()
hold_data_in_memory_after_access_strategy = HoldDataInMemoryAfterAccessStrategy()